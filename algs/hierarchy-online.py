from bloom_filter2 import BloomFilter
from collections import defaultdict
import getopt
import sys
import os
import torch
import torch.nn as nn
from enum import Enum

from lru import LRU


FEATURE_PATH = "/home/janechen/cache/output/features/test-set-real"

# request length of each stage
WARMUP_LENGTH = 1000000
FEATURE_COLLECTION_MAX_LENGTH = 1000000
BANDIT_ROUND_LENGTH = 1000000


ALPHA = 0.001 # defines dc hit benefit
FREQ_OBSERVE_WINDOW = 1000000

# expert correlation neural network hyper parameters
INPUT_SIZE = 22
HIDDEN_SIZE = 15
OUTPUT_SIZE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OnlineStage(Enum):
    CACHE_WARMUP = 0
    FEATURE_COLLECTION = 1
    EXPERT_BANDIT = 2

def parseInput():

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    
    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 't:h:d:', ['trace_dir_path', 'hoc_size', 'dc_size'])
        # Check if the options' length is 3
        if len(opts) != 3:
            print('usage: hierarchy.py -t <trace_dir_path> -h <HOC_size> -d <DC_size>')
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == '-t':
                    trace_path = arg
                if opt == '-h':
                    hoc_s = int(arg)
                if opt == '-d':
                    dc_s = int(arg)               

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print('usage: hierarchy.py -t <trace_path> -f <frequency_threshold> -s <size_threshold> -h <HOC_size> -d <DC_size>')
        sys.exit(2)
        
    return trace_path, hoc_s, dc_s


class OnlineHierarchy:
    def __init__(self, default_freq_thres, default_size_thres, hoc_s, dc_s):
        self.freq_thres = default_freq_thres
        self.size_thres = default_size_thres
        self.hoc_s = hoc_s
        self.dc_s = dc_s
        self.expert_list = []
        for f in [2, 4, 5, 7]:
            for s in [50, 100, 200, 500, 1000]:
                self.expert_list.append('f'+str(f)+'s'+str(s))
                
        self.dc = LRU(dc_s, {})
        self.hoc = LRU(hoc_s, {})
        self.dcAccessTab = defaultdict(list) # dc access table for objects within size threshold {id: access timestamps}
        self.bloom = BloomFilter(max_elements=1000000, error_rate=0.1)
        self.current_stage = OnlineStage.CACHE_WARMUP
        self.stage_parsed_requests = 0
        self.hoc_full = False
        
        self.potential_experts = []
        
        # features
        self.iat_avg = []
        self.sd_avg = []
        self.size_avg = 0
        self.edc_avg = []
        
        # bandit states
        self.round = 0
        self.round_request_num = 0
        
        # statistics
        self.tot_req_num = 0
        self.tot_hoc_hit = 0
        self.tot_obj_hit = 0
        self.tot_req_bytes = 0
        self.tot_byte_miss = 0
        self.tot_disk_read = 0
        self.tot_disk_write = 0
        
    
    def loadModels(self):
        class NeuralNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size) 
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, num_classes)  
            
            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out
        
        self.models = {}
        for exp0 in self.expert_list:
            for exp1 in self.expert_list:
                if exp0 != exp1:
                    model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
                    model.load_state_dict(torch.load(os.path.join("../cache/output/models/", exp0+"-"+exp1, "model-h"+str(HIDDEN_SIZE)+".ckpt")))
                    self.models[(exp0, exp1)] = model
    
    def checkFeatureConfidence(self):
        if self.stage_parsed_requests == FEATURE_COLLECTION_MAX_LENGTH:
            return True
        # TODO: add feature collection confidence check
        return False
    
    def extractThres(self, expert):
        freq_thres = int(expert.split('s')[0].replace('f', ''))
        size_thres = int(expert.split('s')[1])
        return freq_thres, size_thres
    
    def bandit(self):
        
        return
        
    def stageTransition(self):
        if self.current_stage == OnlineStage.CACHE_WARMUP and self.stage_parsed_requests == WARMUP_LENGTH:
            assert self.hoc_full
            self.current_stage = OnlineStage.FEATURE_COLLECTION
            self.stage_parsed_requests = 0
        if self.current_stage == OnlineStage.FEATURE_COLLECTION and self.checkFeatureConfidence():
            # TODO: replace feature load with online feature collection
            self.
            self.current_stage = OnlineStage.EXPERT_BANDIT
            self.stage_parsed_requests = 0
        if self.current_stage == OnlineStage.EXPERT_BANDIT:
            self.current_stage = OnlineStage.CACHE_WARMUP
            self.stage_parsed_requests = 0
        return
    
    # promote an object from dc to hoc
    def promote(self, id, size):
        # delete the object from dc and the dc access table
        self.dc.removeFromCache(id, size)
        del self.dcAccessTab[id]

        # when hoc is full, demote the hoc's lru object to the dc
        while self.hoc.current_size + size > self.hoc.cache_size:
            if not self.hoc_full:
                print("HOC full at request {:d}".format(self.tot_req_num))
                sys.stdout.flush()
            self.demote()

        # add the object to hoc
        self.hoc.addToCache(id, size)

    # demote the lru object from hoc to dc
    def demote(self):
        self.hoc_full = True

        # evict the lru object from hoc
        id, size = self.hoc.evict()

        # add the object to dc
        evicted = self.dc.miss(id, size)
        disk_write += size/4

        # remove the evicted objects in access table
        for evicted_id in evicted:
            if evicted_id in self.dcAccessTab:
                del self.dcAccessTab[evicted_id]

    def countFreq(self, current_t, id):
        for t in self.dcAccessTab[id]:
            if current_t - t > FREQ_OBSERVE_WINDOW:
                self.dcAccessTab[id].remove(t)
            else:
                break
        return len(self.dcAccessTab[id])
    
    def request(self, t, id, size):
        obj_hit = byte_miss = disk_read = disk_write = 0

        if id in self.hoc:
            self.hoc.hit(id)
            obj_hit = 1
        elif id in self.dc:
            self.dc.hit(id)
            disk_read += size/4
            if size < self.size_thres:
                self.dcAccessTab[id].append(t)
                if self.countFreq(t, id) == self.freq_thres:
                    self.promote(id, size)

            obj_hit = ALPHA
        else:
            evicted = self.dc.miss(id, size)
            disk_write += size/4
            for evicted_id in evicted:
                if evicted_id in self.dcAccessTab:
                    del self.dcAccessTab[evicted_id]

            if size < self.size_thres:
                self.dcAccessTab[id].append(t)
            byte_miss = size

        return obj_hit, byte_miss, disk_read, disk_write
    
    def collectStat(self, obj_hit, byte_miss, disk_read, disk_write, size):
        self.tot_byte_miss += byte_miss
        self.tot_disk_read += disk_read
        self.tot_disk_write += disk_write
        self.tot_obj_hit += obj_hit
        if obj_hit == 1:
            self.tot_hoc_hit += 1
        self.tot_req_num += 1
        self.tot_req_bytes += size
    
    def feedRequest(self, t, id, size):
        if id in self.bloom:
            obj_hit, byte_miss, disk_read, disk_write = self.request(t, id, size)
        else:
            obj_hit = disk_read = disk_write = 0
            byte_miss = size
        self.bloom.add(id)
        if self.current_stage != OnlineStage.CACHE_WARMUP:
            self.collectStat(obj_hit, byte_miss, disk_read, disk_write, size)
        if self.current_stage == OnlineStage.FEATURE_COLLECTION:
            # TODO: collect features online
            pass
        self.stage_parsed_requests += 1
        self.stageTransition()
        return


def main():
    
    trace_path, hoc_s, dc_s = parseInput()
    
    if None not in (trace_path, hoc_s, dc_s):
        print('trace: {}, HOC size: {}, DC size: {}'.format(trace_path, hoc_s, dc_s))
    else:
        sys.exit(2)
    
    name = trace_path.split('/')[-1].split('.')[0]
    cache = OnlineHierarchy(4, 50, hoc_s, dc_s)
    
    for line in open(trace_path):
        line = line.split(',')
        t = int(line[0])
        id = int(line[1])
        size = int(line[2])
        cache.feedRequest(t, id, size)
    
    print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:d}, disk write: {:d}'.format(cache.tot_hoc_hit/cache.tot_req_num*100, cache.tot_obj_hit/cache.tot_req_num*100, cache.tot_byte_miss/cache.tot_req_bytes*100, cache.tot_disk_read, cache.tot_disk_write))
    return 0

if __name__ == '__main__':
    main()
