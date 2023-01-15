import pickle
from bloom_filter2 import BloomFilter
from collections import defaultdict
import getopt
import sys
import time
import os
import numpy as np

from lru import LRU

WARMUP_LENGTH = 1000000

trace_path = output_dir = freq_thres = size_thres = hoc_s = dc_s = freq_percentile = size_percentile = collection_length = None
alpha = 0.001 # defines dc hit benefit
t_inter = 1000000
bloom_miss = compulsory_miss = capacity_miss = admission_miss = 0

def parseInput():

    global trace_path, output_dir, hoc_s, dc_s, freq_percentile, size_percentile, collection_length

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    
    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 't:o:f:s:h:d:l:', ['trace_path', 'output_dir', 'freq_percentile', 'size_percentile', 'hoc_size', 'dc_size', 'collection_length'])
        # Check if the options' length is 3
        if len(opts) != 7:
            print('usage: percentile.py -t <trace_path> -o <output_dir> -f <freq_percentile> -s <size_percentile> -h <HOC_size> -d <DC_size> -l <collection_length>')
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == '-t':
                    trace_path = arg
                if opt == '-o':
                    output_dir = arg
                if opt == '-f':
                    freq_percentile = int(arg)
                if opt == '-s':
                    size_percentile = int(arg)
                if opt == '-h':
                    hoc_s = int(arg)
                if opt == '-d':
                    dc_s = int(arg)       
                if opt == '-l':
                    collection_length = int(arg)        

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print('usage: percentile.py -t <trace_path> -o <output_dir> -f <freq_percentile> -s <size_percentile> -h <HOC_size> -d <DC_size> -l <collection_length>')
        sys.exit(2)

def request(t, id, size):
    obj_hit = 0
    byte_miss = 0
    global tot_num, compulsory_miss, admission_miss, capacity_miss, disk_write

    if id in hoc:
        hoc.hit(id)
        obj_hit = 1
    elif id in dc:
        if tot_num >= WARMUP_LENGTH:
            if id not in hoc_set:
                admission_miss += 1
            else:
                capacity_miss += 1
        dc.hit(id)
        global disk_read
        if tot_num >= WARMUP_LENGTH:
            disk_read += size/4
        if size < size_thres:
            dcAccessTab[id].append(t)
            if countFreq(id) == freq_thres:
                promote(id, size)

        obj_hit = alpha
        if tot_num >= WARMUP_LENGTH:
            disk_write += size/4
    else:
        evicted = dc.miss(id, size)
        if tot_num >= WARMUP_LENGTH:
            compulsory_miss += 1

        if tot_num >= WARMUP_LENGTH:
            disk_write += size/4
        global dc_uniq_size
        if id not in dc_set:
            dc_set.add(id)
            dc_uniq_size += size
        for evicted_id in evicted:
            if evicted_id in dcAccessTab:
                del dcAccessTab[evicted_id]

        if size < size_thres:
            dcAccessTab[id].append(t)
        byte_miss = size

    return obj_hit, byte_miss

# promote an object from dc to hoc
def promote(id, size):
    # delete the object from dc and the dc access table
    dc.removeFromCache(id, size)
    del dcAccessTab[id]

    # when hoc is full, demote the hoc's lru object to the dc
    while hoc.current_size + size > hoc.cache_size:
        demote()

    # add the object to hoc
    hoc.addToCache(id, size)

    if id not in hoc_set:
        global hoc_uniq_size
        hoc_set.add(id)
        hoc_uniq_size += size

# demote the lru object from hoc to dc
def demote():
    global isWarmup
    isWarmup = False

    # evict the lru object from hoc
    id, size = hoc.evict()

    # add the object to dc
    evicted = dc.miss(id, size)
    global disk_write
    global tot_num
    if tot_num >= WARMUP_LENGTH:
        disk_write += size/4

    if id not in dc_set:
        dc_set.add(id)
        global dc_uniq_size
        dc_uniq_size += size

    # remove the evicted objects in access table
    for evicted_id in evicted:
        if evicted_id in dcAccessTab:
            del dcAccessTab[evicted_id]

def countFreq(id):
    for t in dcAccessTab[id]:
        if currentT - t > t_inter:
            dcAccessTab[id].remove(t)
        else:
            break
    return len(dcAccessTab[id])

def run():

    global currentT, hoc, dc, dcAccessTab, bloom, dc_set, hoc_set, dc_uniq_size, hoc_uniq_size, disk_read, disk_write, freq_percentile, size_percentile, collection_length, freq_thres, size_thres
    freq_thres = 2
    size_thres = 20
    dc_uniq_size = hoc_uniq_size = tot_onehit_obj = disk_read = disk_write = 0
    
    dc_set = set()
    hoc_set = set()

    dc = LRU(dc_s, {})
    hoc = LRU(hoc_s, {})
    dcAccessTab = defaultdict(list) # dc access table for objects within size threshold {id: access timestamps}
    bloom = BloomFilter(max_elements=1000000, error_rate=0.1)

    global tot_num, bloom_miss, compulsory_miss, admission_miss, capacity_miss
    tot_num = tot_req = tot_bytes = tot_obj_hit = tot_byte_miss = tot_hoc_hit = 0

    global isWarmup
    isWarmup = True
    firstWarmup = True
    # reqs = []
    # freqs = []
    # hits = []
    obj_count = {}
    size_list = []
    
    # print(os.path.join(output_dir, 'f'+str(freq_thres)+'s'+str(size_thres)+"-hits.pkl"))

    with open(trace_path) as fp:
        for line in fp:
            line = line.split(',')
            t = int(line[0])
            id = int(line[1])
            size = int(line[2])
            currentT = t
            if id not in obj_count.keys():
                obj_count[id] = 0
            obj_count[id] += 1
            size_list.append(size)
            if id in bloom:
                obj_hit, byte_miss = request(t, id, size)
            else:
                tot_onehit_obj += 1
                obj_hit = 0
                byte_miss = size
                if tot_num >= WARMUP_LENGTH:
                    bloom_miss += 1
            bloom.add(id)

            if tot_num >= WARMUP_LENGTH:
            # if not isWarmup:
            #     if firstWarmup:
            #         print("Warmup done after request {:d}".format(tot_num))
            #         firstWarmup = False
                if obj_hit == 1:
                    tot_hoc_hit += 1
                    # hits.append(1)
                # else:
                    # hits.append(0)
                tot_obj_hit += obj_hit
                tot_byte_miss += byte_miss
                tot_req += 1
                tot_bytes += size 
            tot_num += 1
            if tot_num > WARMUP_LENGTH and tot_num % collection_length == 0:
                freq_thres = np.percentile(list(obj_count.values()), freq_percentile)
                size_thres = np.percentile(size_list, size_percentile)
                print(freq_percentile, freq_thres, size_percentile, size_thres)
                sys.stdout.flush()
            #     print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:.4f}, disk write: {:.4f}'.format(tot_hoc_hit/tot_req*100, tot_obj_hit/tot_req*100, tot_byte_miss/tot_bytes*100, disk_read, disk_write))
            #     print('tot hoc size: {:d}, tot dc size: {:d}, one hit obj num: {:d}'.format(hoc_uniq_size, dc_uniq_size, tot_onehit_obj))
            #     print('bloom_miss: {:d}, compulsory_miss: {:d}, admission_miss: {:d}, capacity_miss: {:d}'.format(bloom_miss, compulsory_miss, admission_miss, capacity_miss))
            #     sys.stdout.flush()
                
            #     pickle.dump(hits, open(os.path.join("../cache/output", trace_path.split('/')[6].split('.')[0], 'f'+str(freq_thres)+'s'+str(size_thres)+"-hits.pkl"), "wb"))

                # import numpy as np
                # import matplotlib.pyplot as plt

                # plt.figure(figsize=(100,100))
                # plt.scatter(reqs, ids, s=0.001)
                # # plt.xlim([0, 2000000])
                # plt.show()
                # plt.savefig("./fig/"+trace_path[35:39]+"f"+str(freq_thres)+"s"+str(size_thres)+"-request-cacheids.png")
                # tot_req = tot_bytes = tot_obj_hit = tot_byte_miss = tot_hoc_hit = dc_uniq_size = hoc_uniq_size = tot_onehit_obj = disk_read = disk_write = 0
                # dc.clear()
                # hoc.clear()
                # dcAccessTab.clear()
                # del bloom
                # bloom = BloomFilter(max_elements=1000000, error_rate=0.1)
                # break
        print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:.4f}, disk write: {:.4f}'.format(tot_hoc_hit/tot_req*100, tot_obj_hit/tot_req*100, tot_byte_miss/tot_bytes*100, disk_read, disk_write))
        print('tot hoc size: {:d}, tot dc size: {:d}, one hit obj num: {:d}'.format(hoc_uniq_size, dc_uniq_size, tot_onehit_obj))
        print('bloom_miss: {:d}, compulsory_miss: {:d}, admission_miss: {:d}, capacity_miss: {:d}'.format(bloom_miss, compulsory_miss, admission_miss, capacity_miss))
        sys.stdout.flush()
        
        # pickle.dump(hits, open(os.path.join(output_dir, 'f'+str(freq_thres)+'s'+str(size_thres)+"-hits.pkl"), "wb"))


def main():
    parseInput()
    if None not in (trace_path, freq_percentile, size_percentile, hoc_s, dc_s):
        print('trace: {}, freq: {}, size: {}, HOC size: {}, DC size: {}'.format(trace_path, freq_thres, size_thres, hoc_s, dc_s))
    else:
        sys.exit(2)

    run()

    return 0

if __name__ == '__main__':
    main()
