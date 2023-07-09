import pickle
from bloom_filter2 import BloomFilter
from collections import defaultdict
import getopt
import sys
import time
import os

from lru import LRU

WARMUP_LENGTH = 1000000
ROUND_LEN = 500000
BASE_DIR = "/scratch1/09498/janechen/mydata/"

trace_path = output_dir = freq_thres1 = size_thres1 = freq_thres2 = size_thres2 = hoc_s = dc_s = None
alpha = 0.001  # defines dc hit benefit
t_inter = 1000000


def parseInput():

    global trace_path, output_dir, freq_thres, size_thres, hoc_s, dc_s

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]

    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 't:o:f:s:g:u:h:d:', [
                                   'trace_path', 'output_dir', 'freq_thres1', 'size_thres1', 'freq_thres2', 'size_thres2', 'hoc_size', 'dc_size'])
        # Check if the options' length is 3
        if len(opts) != 8:
            print('usage: hierarchy.py -t <trace_path> -o <output_dir> -f <frequency_threshold1> -s <size_threshold1> -g <frequency_threshold2> -u <size_threshold2> -h <HOC_size> -d <DC_size>')
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == '-t':
                    trace_path = arg
                if opt == '-o':
                    output_dir = arg
                if opt == '-f':
                    freq_thres1 = int(arg)
                if opt == '-s':
                    size_thres1 = int(arg)
                if opt == '-g':
                    freq_thres2 = int(arg)
                if opt == '-u':
                    size_thres2 = int(arg)
                if opt == '-h':
                    hoc_s = int(arg)
                if opt == '-d':
                    dc_s = int(arg)

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print('usage: hierarchy.py -t <trace_path> -f <frequency_threshold> -s <size_threshold> -h <HOC_size> -d <DC_size>')
        sys.exit(2)


class OfflineRun():
    def __init__(self, freq_thres, size_thres, hoc_s, dc_s):
        self.name = "f"+str(freq_thres)+"s"+str(size_thres)
        self.freq_thres = freq_thres
        self.size_thres = size_thres
        self.hoc_s = hoc_s
        self.dc_s = dc_s

        self.dc = LRU(dc_s, {})
        self.hoc = LRU(hoc_s, {})
        # dc access table for objects within size threshold {id: access timestamps}
        self.dcAccessTab = defaultdict(list)
        
        self.warmup_complete = False
        
        self.tot_req = self.tot_hoc_hit = self.tot_hoc_byte_hit = self.tot_obj_hit = self.tot_byte_miss = self.tot_bytes = self.tot_onehit_obj = self.disk_read = self.disk_write = 0

    def addMiss(self, size):
        if self.warmup_complete:
            self.tot_req += 1
            self.tot_bytes += size
            self.tot_byte_miss += size
    
    def request(self, t, id, size):
        hoc_hit = hoc_byte_hit = obj_hit = byte_miss = disk_read = disk_write = 0

        if id in self.hoc:
            self.hoc.hit(id)
            obj_hit = 1
            hoc_hit = 1
            hoc_byte_hit = size
        elif id in self.dc:
            self.dc.hit(id)
            disk_read = size/4
            if size < self.size_thres:
                self.dcAccessTab[id].append(t)
                if self.countFreq(id) == self.freq_thres:
                    self.promote(id, size)
            obj_hit = alpha
        else:
            evicted = self.dc.miss(id, size)
            disk_write += size/4
            for evicted_id in evicted:
                if evicted_id in self.dcAccessTab:
                    del self.dcAccessTab[evicted_id]

            if size < self.size_thres:
                self.dcAccessTab[id].append(t)
                if self.countFreq(t, id) == self.freq_thres:
                    disk_write += self.promote(id, size)
            byte_miss = size
        
        if self.warmup_complete:
            self.tot_req += 1
            self.tot_bytes += size
            self.tot_byte_miss += byte_miss
            self.tot_obj_hit += obj_hit
            self.tot_hoc_hit += hoc_hit
            self.tot_hoc_byte_hit += hoc_byte_hit
            self.disk_read += disk_read
            self.disk_write += disk_write

        return hoc_hit

    # promote an object from dc to hoc
    def promote(self, id, size):
        # delete the object from dc and the dc access table
        self.dc.removeFromCache(id, size)
        del self.dcAccessTab[id]
        
        disk_write = 0

        # when hoc is full, demote the hoc's lru object to the dc
        while self.hoc.current_size + size > self.hoc.cache_size:
            disk_write += self.demote()

        # add the object to hoc
        self.hoc.addToCache(id, size)
        
        return disk_write

    # demote the lru object from hoc to dc
    def demote(self):

        # evict the lru object from hoc
        id, size = self.hoc.evict()

        # add the object to dc
        evicted = self.dc.miss(id, size)
        disk_write = size/4

        # remove the evicted objects in access table
        for evicted_id in evicted:
            if evicted_id in self.dcAccessTab:
                del self.dcAccessTab[evicted_id]
        
        return disk_write

    def countFreq(self, current_t, id):
        for t in self.dcAccessTab[id]:
            if current_t - t > t_inter:
                self.dcAccessTab[id].remove(t)
            else:
                break
        return len(self.dcAccessTab[id])


def run():
    trace_name = trace_path.split('/')[-1].split('.')[0]

    size_table = {}  # id: size
    bloom = BloomFilter(max_elements=1000000, error_rate=0.1)
    run_exp1 = OfflineRun(freq_thres1, size_thres1, hoc_s, dc_s)
    run_exp2 = OfflineRun(freq_thres2, size_thres2, hoc_s, dc_s)
    
    tot_num = count = 0
    inputs = []
    labels = []
    bucket_list = [10, 20, 50, 100, 500, 1000, 5000]
    
    hit_hit_prob = 0 # pi(e1_hit | e0_hit)
    e0_hit_count = 0
    e0_miss_count = 0
    hit_miss_prob = 0 # pi(e1_hit | e0_miss)
    bucket_count = [0]*(len(bucket_list)+1)
    
    feature_set = ['iat_avg', 'sd_avg', 'size_avg']
    
    feature = []
        
    features = pickle.load(open(os.path.join(BASE_DIR, "features/", trace_name, "3M.pkl"), "rb"))
    for f in feature_set:
        v = features[f]
        if type(v) is dict or type(v) is defaultdict:
            values = [value for key,value in sorted(v.items())]   
            feature += values
        else:
            feature.append(v)
    
    assert len(feature) == 15

    with open(trace_path, 'r') as fp:
        for line in fp:
            line = line.split(',')
            t = int(line[0])
            id = int(line[1])
            if id in size_table.keys():
                size = size_table[id]
            else:
                size = int(line[2])
                size_table[id] = size
                
            if id in bloom:
                obj_hit1 = run_exp1.request(t, id, size)
                obj_hit2 = run_exp2.request(t, id, size)
            else:
                run_exp1.addMiss(size)
                run_exp2.addMiss(size)
            bloom.add(id)
            
            if tot_num > WARMUP_LENGTH:
                if tot_num % ROUND_LEN == 0:
                    hit_hit_prob = hit_hit_prob/e0_hit_count
                    hit_miss_prob = hit_miss_prob/e0_miss_count
                    input = []
                    input.extend(feature)
                    input.extend(bucket_count)
                    inputs.append([input, e0_hit_count, e0_miss_count])
                    labels.append([hit_hit_prob, hit_miss_prob])
                    e0_hit_count = e0_miss_count = hit_hit_prob = hit_miss_prob = 0
                    bucket_count = [0]*(len(bucket_list)+1)
                if obj_hit1 == 1:
                    e0_hit_count += 1
                    if obj_hit2 == 1:
                        hit_hit_prob += 1
                else:
                    e0_miss_count += 1
                    if obj_hit2 == 1:
                        hit_miss_prob += 1
                        
                for j in range(len(bucket_list)):
                    if size < bucket_list[j]:
                        bucket_count[j] += 1
                        break
                if size >= bucket_list[-1]:
                    bucket_count[-1] += 1
            
            tot_num += 1
            if tot_num == WARMUP_LENGTH:
                run_exp1.warmup_complete = True
                run_exp2.warmup_complete = True
                
        if tot_num % ROUND_LEN == 0:
            hit_hit_prob = hit_hit_prob/e0_hit_count
            hit_miss_prob = hit_miss_prob/e0_miss_count
            input = []
            input.extend(feature)
            input.extend(bucket_count)
            inputs.append([input, e0_hit_count, e0_miss_count])
            labels.append([hit_hit_prob, hit_miss_prob])
            
        pickle.dump(inputs, open(os.path.join(BASE_DIR, "correlations", "f"+str(freq_thres1)+"s"+str(size_thres1)+"-"+"f"+str(freq_thres2)+"s"+str(size_thres2), trace_name+"-input.pkl"), "wb"))
        pickle.dump(labels, open(os.path.join(BASE_DIR, "correlations", "f"+str(freq_thres1)+"s"+str(size_thres1)+"-"+"f"+str(freq_thres2)+"s"+str(size_thres2), trace_name+"-labels.pkl"), "wb"))

        print('trace: {}, f: {}, s: {}, final hoc hit: {:.4f}%, hoc byte miss: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:.4f}, disk write: {:.4f}'.format(trace_name, freq_thres1, size_thres1, run_exp1.tot_hoc_hit/run_exp1.tot_req*100, (run_exp1.tot_bytes-run_exp1.tot_hoc_byte_hit)/run_exp1.tot_bytes*100, run_exp1.tot_obj_hit/run_exp1.tot_req*100, run_exp1.tot_byte_miss/run_exp1.tot_bytes*100, run_exp1.disk_read, run_exp1.disk_write))
        print('trace: {}, f: {}, s: {}, final hoc hit: {:.4f}%, hoc byte miss: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:.4f}, disk write: {:.4f}'.format(trace_name, freq_thres2, size_thres2, run_exp2.tot_hoc_hit/run_exp2.tot_req*100, (run_exp2.tot_bytes-run_exp2.tot_hoc_byte_hit)/run_exp2.tot_bytes*100, run_exp2.tot_obj_hit/run_exp2.tot_req*100, run_exp2.tot_byte_miss/run_exp2.tot_bytes*100, run_exp2.disk_read, run_exp2.disk_write))
        sys.stdout.flush()


def main():
    parseInput()
    if None not in (trace_path, freq_thres1, size_thres1, freq_thres2, size_thres2, hoc_s, dc_s):
        print('trace: {}, freq: {}, size: {}, HOC size: {}, DC size: {}'.format(
            trace_path, freq_thres, size_thres, hoc_s, dc_s))
    else:
        sys.exit(2)

    run()

    return 0


if __name__ == '__main__':
    main()
