import pickle
from bloom_filter2 import BloomFilter
from collections import defaultdict
import getopt
import sys
import time
import os
import numpy as np
import copy

from lru import LRU

WARMUP_LENGTH = 1000000

trace_path = output_dir = freq_thres = size_thres = hoc_s = dc_s = collection_length = None
alpha = 0.001 # defines dc hit benefit
t_inter = 1000000

frequency_list = [2, 3, 4, 5, 6, 7]
size_list = [10, 20, 50, 100, 500, 1000]

def parseInput():

    global trace_path, output_dir, hoc_s, dc_s, collection_length

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    
    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 't:o:f:s:h:d:l:', ['trace_path', 'output_dir', 'hoc_size', 'dc_size', 'collection_length'])
        # Check if the options' length is 3
        if len(opts) != 5:
            print('usage: hillclimbing.py -t <trace_path> -o <output_dir> -h <HOC_size> -d <DC_size> -l <collection_length>')
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == '-t':
                    trace_path = arg
                if opt == '-o':
                    output_dir = arg
                if opt == '-h':
                    hoc_s = int(arg)
                if opt == '-d':
                    dc_s = int(arg)       
                if opt == '-l':
                    collection_length = int(arg)        

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print('usage: hillclimbing.py -t <trace_path> -o <output_dir> -h <HOC_size> -d <DC_size> -l <collection_length>')
        sys.exit(2)

class Cache:
    def __init__(self, hoc_s, dc_s, freq_thres, size_thres):
        self.dc = LRU(dc_s, {})
        self.hoc = LRU(hoc_s, {})
        self.dcAccessTab = defaultdict(list) # dc access table for objects within size threshold {id: access timestamps}
        self.bloom = BloomFilter(max_elements=1000000, error_rate=0.1)
        
        self.freq_thres = freq_thres
        self.size_thres = size_thres
        
        # record period of stats for this cache
        self.obj_hit = 0
        # self.byte_miss = 0
        self.disk_write = 0
    
    def copy_state(self, cache):
        # fix this by using iteration to do deepcopy
        self.dc.copy_state(cache.dc)
        self.hoc.copy_state(cache.hoc)
        self.dcAccessTab = copy.deepcopy(cache.dcAccessTab)
    
    def reset(self, new_freq_thres, new_size_thres):
        self.freq_thres = new_freq_thres
        self.size_thres = new_size_thres

        self.obj_hit = self.disk_write = 0

        
    def request(self, t, id, size):
        obj_hit = 0
        disk_write = 0
        
        if id in self.bloom:
            global tot_num

            if id in self.hoc:
                self.hoc.hit(id)
                obj_hit = 1
            elif id in self.dc:
                self.dc.hit(id)
                if size < self.size_thres:
                    self.dcAccessTab[id].append(t)
                    if self.countFreq(id) == self.freq_thres:
                        self.promote(id, size)

                obj_hit = alpha
                if tot_num >= WARMUP_LENGTH:
                    disk_write += size/4
            else:
                evicted = self.dc.miss(id, size)

                if tot_num >= WARMUP_LENGTH:
                    disk_write += size/4
                for evicted_id in evicted:
                    if evicted_id in self.dcAccessTab:
                        del self.dcAccessTab[evicted_id]

                if size < self.size_thres:
                    self.dcAccessTab[id].append(t)
        
        self.bloom.add(id)
        self.obj_hit += obj_hit
        self.disk_write += disk_write

        return obj_hit, disk_write

    # promote an object from dc to hoc
    def promote(self, id, size):
        # delete the object from dc and the dc access table
        self.dc.removeFromCache(id, size)
        del self.dcAccessTab[id]

        # when hoc is full, demote the hoc's lru object to the dc
        while self.hoc.current_size + size > self.hoc.cache_size:
            self.demote()

        # add the object to hoc
        self.hoc.addToCache(id, size)

    # demote the lru object from hoc to dc
    def demote(self):
        global isWarmup
        isWarmup = False

        # evict the lru object from hoc
        id, size = self.hoc.evict()

        # add the object to dc
        evicted = self.dc.miss(id, size)
        # global disk_write
        # global tot_num
        # if tot_num >= WARMUP_LENGTH:
        #     disk_write += size/4

        # if id not in dc_set:
        #     dc_set.add(id)
        #     global dc_uniq_size
        #     dc_uniq_size += size

        # remove the evicted objects in access table
        for evicted_id in evicted:
            if evicted_id in self.dcAccessTab:
                del self.dcAccessTab[evicted_id]

    def countFreq(self, id):
        for t in self.dcAccessTab[id]:
            if currentT - t > t_inter:
                self.dcAccessTab[id].remove(t)
            else:
                break
        return len(self.dcAccessTab[id])

def run():

    global currentT, disk_write, collection_length, freq_thres, size_thres

    # hoc_s = 100
    # dc_s = 10000
    f_cache_i = s_cache_i = 0
    f_shadow_i = s_shadow_i = 1
    real_cache = Cache(hoc_s, dc_s, frequency_list[f_cache_i], size_list[s_cache_i])
    f_shadow_cache = Cache(hoc_s, dc_s, frequency_list[f_shadow_i], size_list[s_cache_i]) # shadow cache for tuning frequency threshold
    s_shadow_cache = Cache(hoc_s, dc_s, frequency_list[f_cache_i], size_list[s_shadow_i]) # shadow cache for tuning size threshold
    
    global tot_num
    tot_num = tot_req = tot_bytes = tot_obj_hit = tot_byte_miss = tot_hoc_hit = tot_disk_write = 0

    # real_cache.request(0, 0, 1)
    # real_cache.request(0, 0, 1)
    # print(real_cache.dcAccessTab)
    # f_shadow_cache.copy_state(real_cache)
    # f_shadow_cache.request(1, 1, 2)
    # f_shadow_cache.request(1, 1, 2)
    # real_cache.request(2, 2, 3)
    # real_cache.request(2, 2, 3)
    # print(id(real_cache.dc.lru.htbl))
    # print(real_cache.dc.lru.htbl)
    # print(id(f_shadow_cache.dc.lru.htbl))
    # print(f_shadow_cache.dc.lru.htbl)

    global isWarmup
    isWarmup = True

    with open(trace_path) as fp:
        for line in fp:
            try:
                line = line.split(',')
                t = int(line[0])
                id = int(line[1])
                size = int(line[2])
                currentT = t

            except:
                print(trace_path, line, file=sys.stderr)
                continue

            obj_hit, disk_write = real_cache.request(t, id, size)
            f_shadow_cache.request(t, id, size)
            s_shadow_cache.request(t, id, size)

            if tot_num >= WARMUP_LENGTH:
                if obj_hit == 1:
                    tot_hoc_hit += 1
                tot_obj_hit += obj_hit
                tot_disk_write += disk_write
                tot_req += 1
                tot_bytes += size 
            tot_num += 1
            if tot_num > WARMUP_LENGTH and tot_num % collection_length == 0:
                print('real cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(real_cache.obj_hit/collection_length*100, real_cache.disk_write))
                print('f shadow cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(f_shadow_cache.obj_hit/collection_length*100, f_shadow_cache.disk_write))
                print('s shadow cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(s_shadow_cache.obj_hit/collection_length*100, s_shadow_cache.disk_write))
                print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk write: {:.4f}'.format(tot_hoc_hit/tot_req*100, tot_obj_hit/tot_req*100, tot_byte_miss/tot_bytes*100, tot_disk_write))
                sys.stdout.flush()
                
                new_f_i = 0
                new_s_i = 0
                if real_cache.obj_hit >= f_shadow_cache.obj_hit:
                    new_f_i = f_cache_i
                else:
                    new_f_i = f_shadow_i
                if real_cache.obj_hit >= s_shadow_cache.obj_hit:
                    new_s_i = s_cache_i
                else:
                    new_s_i = s_shadow_i
                
                if new_f_i == max(f_cache_i, f_shadow_i):
                    if new_f_i+1 < len(frequency_list):
                        f_shadow_i = new_f_i+1
                    else:
                        f_shadow_i = min(f_cache_i, f_shadow_i) # real cache already the largest value, shadow cache choose the other value
                else:
                    if new_f_i-1 >= 0:
                        f_shadow_i = new_f_i-1
                    else:
                        f_shadow_i = max(f_cache_i, f_shadow_i)

                if new_s_i == max(s_cache_i, s_shadow_i):
                    if new_s_i+1 < len(size_list):
                        s_shadow_i = new_s_i+1
                    else:
                        s_shadow_i = min(s_cache_i, s_shadow_i) # real cache already the largest value, shadow cache choose the other value
                else:
                    if new_s_i-1 >= 0:
                        s_shadow_i = new_s_i-1
                    else:
                        s_shadow_i = max(s_cache_i, s_shadow_i)
                f_cache_i = new_f_i
                s_cache_i = new_s_i
                f_shadow_cache.copy_state(real_cache)
                s_shadow_cache.copy_state(real_cache)
                real_cache.reset(frequency_list[f_cache_i], size_list[s_cache_i])
                f_shadow_cache.reset(frequency_list[f_shadow_i], size_list[s_cache_i])
                s_shadow_cache.reset(frequency_list[f_cache_i], size_list[s_shadow_i])
                
                print('f_cache_thres: {:d}, s_cache_thres: {:d}, f_shadow_thres: {:d}, s_cache_thres: {:d}'.format(frequency_list[f_cache_i], size_list[s_cache_i], frequency_list[f_shadow_i], size_list[s_shadow_i]))
                sys.stdout.flush()
                
    print('real cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(real_cache.obj_hit/collection_length*100, real_cache.disk_write))
    print('f shadow cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(f_shadow_cache.obj_hit/collection_length*100, f_shadow_cache.disk_write))
    print('s shadow cache stage hoc hit: {:.4f}%, disk write: {:.4f}'.format(s_shadow_cache.obj_hit/collection_length*100, s_shadow_cache.disk_write))
    print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk write: {:.4f}'.format(tot_hoc_hit/tot_req*100, tot_obj_hit/tot_req*100, tot_byte_miss/tot_bytes*100, tot_disk_write))
    sys.stdout.flush()

def main():
    parseInput()
    if None not in (trace_path, hoc_s, dc_s, collection_length):
        print('trace: {}, HOC size: {}, DC size: {}'.format(trace_path, hoc_s, dc_s))
    else:
        sys.exit(2)

    run()

    return 0

if __name__ == '__main__':
    main()
