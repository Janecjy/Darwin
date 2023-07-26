from bloom_filter2 import BloomFilter
from collections import defaultdict
import getopt
import sys
import os
import pickle
import torch
import torch.nn as nn
from enum import Enum
from sklearn.cluster import KMeans
import math
import numpy as np
from pynverse import inversefunc
import random
import time
import datetime

from lru import LRU
from utils.traffic_model.extract_feature import Feature_Cache


# path
CLUSTER_MODEL_PATH =""
CLUSTER_RESULT_PATH = ""
# trained feature parameters
FEATURE_MAX_LIST = None
FEATURE_MIN_LIST = None
# FEATURE_PATH = "/home/janechen/cache/output/features/test-set-real"
# REAL_BEST_PATH = "/home/janechen/cache/output/test_best_result.pkl"
model_path = ""
BEST_THRES = 1

# request length of each stage
WARMUP_LENGTH = 1000000
FEATURE_COLLECTION_MAX_LENGTH = 3000000


ALPHA = 0.001 # defines dc hit benefit
FREQ_OBSERVE_WINDOW = 1000000

# feature parameters
FEATURE_CACHE_SIZE = 10000000
MAX_HIST = 7
IAT_WIN = 50000
SD_WIN = 150000

class OnlineStage(Enum):
    CACHE_WARMUP = 0
    FEATURE_COLLECTION = 1
    BEST_DEPLOYED = 2
        

def parseInput():

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]
    trace_path = model_path = hoc_s = dc_s = None
    
    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(argv, 't:h:d:m:', ['trace_dir_path', 'hoc_size', 'dc_size', 'model_dir_path'])
        # Check if the options' length is 3
        if len(opts) != 4:
            print('usage: hierarchy.py -t <trace_dir_path> -m <model_dir_path> -h <HOC_size> -d <DC_size>')
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == '-t':
                    trace_path = arg
                if opt == '-m':
                    model_path = arg
                if opt == '-h':
                    hoc_s = int(arg)
                if opt == '-d':
                    dc_s = int(arg)               

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print('usage: hierarchy.py -t <trace_dir_path> -m <model_dir_path> -h <HOC_size> -d <DC_size>')
        sys.exit(2)
        
    return trace_path, model_path, hoc_s, dc_s


class OnlineHierarchy:
    def __init__(self, name, default_freq_thres, default_size_thres, hoc_s, dc_s, rate):
        self.name = name
        self.freq_thres = default_freq_thres
        self.size_thres = default_size_thres
        self.hoc_s = hoc_s
        self.dc_s = dc_s
        self.rate = rate
        # self.expert_list = ["f2s50", "f2s100", "f4s50"]
        self.expert_list = []
        for f in [2, 3, 4, 5, 6, 7]:
            for s in [10, 20, 50, 100, 500, 1000]:
                self.expert_list.append('f'+str(f)+'s'+str(s*rate))
        print(self.expert_list)
                
        self.dc = LRU(dc_s, {})
        self.hoc = LRU(hoc_s, {})
        self.dcAccessTab = defaultdict(list) # dc access table for objects within size threshold {id: access timestamps}
        self.bloom = BloomFilter(max_elements=1000000, error_rate=0.1)
        self.current_stage = OnlineStage.CACHE_WARMUP
        self.stage_parsed_requests = 0
        self.hoc_full = False
        
        # features
        self.feature_cache = Feature_Cache(FEATURE_CACHE_SIZE, IAT_WIN, SD_WIN)
        self.initial_objects = list()
        self.initial_times = {}
        self.obj_sizes = defaultdict(int)
        self.obj_reqs = defaultdict(int)
        
        self.obj_iats = defaultdict(list)
        self.iat_avg = dict.fromkeys([x+1 for x in range(MAX_HIST)]) # iat average, iat[k] is the dt average between first arrival and k+1 th arrival
        self.sd_avg = dict.fromkeys([x+1 for x in range(MAX_HIST)]) # sd average, iat[k] is the dt average between first arrival and k+1 th arrival
        self.sd_count = [0]*MAX_HIST
        self.size_avg = 0
        self.size_count = 0
        self.edc_avg = dict.fromkeys([x+1 for x in range(MAX_HIST)])
        for x in self.iat_avg.keys():
            self.iat_avg[x] = 0
            self.sd_avg[x] = 0
            self.edc_avg[x] = 0
        self.feature = []
        
        # statistics
        self.tot_req_num = 0
        self.tot_hoc_hit = 0
        self.tot_obj_hit = 0
        self.tot_req_bytes = 0
        self.tot_byte_miss = 0
        self.tot_disk_read = 0
        self.tot_disk_write = 0
    
    def checkFeatureConfidence(self):
        if self.stage_parsed_requests == FEATURE_COLLECTION_MAX_LENGTH:
            return True
        # TODO: add feature collection confidence check
        return False
    
    def extractThres(self, expert):
        freq_thres = int(expert.split('s')[0].replace('f', ''))
        size_thres = int(expert.split('s')[1])
        return freq_thres, size_thres
    
    def computeFeature(self):
        return
    
    
    def confSort(self, keys):
        return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))
    
    def cluster(self):
        '''Classify trace into cluster to get potential best experts'''
        
        cluster_model = pickle.load(open(CLUSTER_MODEL_PATH, "rb"))
        cluster_result = pickle.load(open(CLUSTER_RESULT_PATH, "rb"))
        
        X_dist = cluster_model.transform([self.feature])[0]
        cluster_index = np.argsort(X_dist)[0]
        expert = cluster_result[cluster_index]
        self.freq_threshold, self.size_threshold = self.extractThres(expert)
        print("cluster predicted best expert:")
        print(expert)
        sys.stdout.flush()
        
        return
    
    def getLatestEstimate(self, e):
        return self.avg_estimated[e]
    
    def calculateW(self, alphas, e):
        result = 0
        for i, ei in enumerate(self.potential_experts):
            if (self.model_variance[(ei, e)]> 0):
                result += alphas[i]/(self.model_variance[(ei, e)])
            else:
                result += alphas[i]/0.00001
        return result
    
    def calculateDelta(self, best_e, e):
        return self.getLatestEstimate(best_e)-self.getLatestEstimate(e)
    
    def findMinConf(self, alphas):
        best_e_index = np.argmax([self.getLatestEstimate(e) for e in self.potential_experts])
        best_e = self.potential_experts[best_e_index]
        w_star = self.calculateW(alphas, best_e)
        results = [w_star*self.calculateW(alphas, e)/(w_star+self.calculateW(alphas, e))*(self.calculateDelta(best_e, e)**2)/2 for e in self.potential_experts if e != best_e]
        return min(results)
    
    def calculateZ(self):
        selected_times_sum = self.round
        normalized_selected_times = [self.selected_times[i]/selected_times_sum for i in self.selected_times.keys()]
        return self.findMinConf(normalized_selected_times)*selected_times_sum
    
    def calculateBeta(self):
        t = self.round
        k = len(self.potential_experts)
        # f = (lambda x: math.exp(k-x)*((x/k)**k))
        beta = k * np.log(t**2+t) + self.f_inverse
        return beta
        
    def calculateEstimated(self, current_e):
        for e in self.potential_experts:
            if (self.model_variance[(current_e, e)] > 0):
                self.estimated_numerator[e] += self.observed_rewards[e][-1]/(self.model_variance[(current_e, e)])
                self.estimated_denominator[e] += 1/(self.model_variance[(current_e, e)])
            else:
                self.estimated_numerator[e] += self.observed_rewards[e][-1]/0.00001
                self.estimated_denominator[e] += 1/0.00001
            self.estimated_rewards[e].append(self.estimated_numerator[e]/self.estimated_denominator[e])
            # update average value of the estimates
            self.avg_estimated[e] = sum(self.estimated_rewards[e])/len(self.estimated_rewards[e])
        return
    
    def selectAlpha(self):
        r_list = [self.findMinConf(alphas) for alphas in self.alpha_list]
        max_index = np.argmax(r_list)
        return self.alpha_list[max_index]
    
    def selectArmWithAlpha(self, alphas):
        arm_index = np.argmax([self.round*alphas[i]-self.selected_times[e] for i, e in enumerate(self.potential_experts)])
        return self.potential_experts[arm_index]
    
    def selectArm(self):
        
        k = len(self.potential_experts)
        if self.round < 3*k:
            # Choose each arm once
            arm = self.potential_experts[self.round%k]
        elif self.round > 5*k:
            arm_index = np.argmax([self.avg_estimated[e] for e in self.potential_experts])
            arm = self.potential_experts[arm_index]
            self.bandit_end = True
        else:
            # if min(list(self.selected_times.values())) <= math.sqrt(self.round):
            #     # select the least selected arm
            #     arm_index = np.argmin(list(self.selected_times.values()))
            #     arm = list(self.selected_times.keys())[arm_index]
            # else:
            # select the arm with highest estimated reward with confidence
            alphas = self.selectAlpha()
            arm = self.selectArmWithAlpha(alphas)
            # arm = "f2s50"
        
        # print(self.round, k)
        # print(arm)
        self.selected_times[arm] += 1
        # set the new freq and size threshold
        new_f, new_s = self.extractThres(arm)
        self.freq_thres = new_f
        self.size_thres = new_s
        
        return
    
    def updateReward(self):
        current_exp = 'f'+str(self.freq_thres)+'s'+str(self.size_thres)
        self.observed_rewards[current_exp].append(self.round_hoc_hit_num/(self.round_request_num/2))
        for e in self.potential_experts:
            if e != current_exp:
                [pred_hit_hit_prob, pred_hit_miss_prob] = self.models[(current_exp, e)]
                e0_hit_count = self.round_hoc_hit_num
                e0_miss_count = self.round_request_num/2 - self.round_hoc_hit_num
                # change to sampling
                pred_e1_hitrate = (np.random.binomial(e0_hit_count, pred_hit_hit_prob) + np.random.binomial(e0_miss_count, pred_hit_miss_prob)) / (e0_hit_count + e0_miss_count)
                # pred_e1_hitrate = (e0_hit_count * pred_hit_hit_prob + e0_miss_count * pred_hit_miss_prob) / (e0_hit_count + e0_miss_count)
                self.observed_rewards[e].append(pred_e1_hitrate)
                model_var = pred_hit_miss_prob*(1-pred_hit_miss_prob)*e0_miss_count/(self.round_request_num/2) + pred_hit_hit_prob*(1-pred_hit_hit_prob)*e0_hit_count/(self.round_request_num/2)
            else:
                model_var = self.round_hoc_hit_num/(self.round_request_num/2)*(1-self.round_hoc_hit_num/(self.round_request_num/2))
            # self.model_variance[(current_exp, e)] = model_var
            self.model_variance_list[(current_exp, e)].append(model_var)
            self.model_variance[(current_exp, e)] = sum(self.model_variance_list[(current_exp, e)])/len(self.model_variance_list[(current_exp, e)])
            print("model variance for ({}, {}): {:.4f}".format(current_exp, e, self.model_variance[(current_exp, e)]))
        
        # update estimated reward
        self.calculateEstimated(current_exp)
        # self.updateBadSet()
        # self.calculateZ()
        # self.selectAlpha()
        print("Round {:d}, request {:d}: selected expert is {}, round hoc hit: {:.4f}%, hoc hit: {:.4f}%".format(self.round, self.tot_req_num, current_exp, self.round_hoc_hit_num_all/self.round_request_num*100, self.tot_hoc_hit/self.tot_req_num*100))
        for e in self.observed_rewards.keys():
            print("Observed reward for {} is {}".format(e, self.observed_rewards[e][-1]))
        print(self.avg_estimated)
        if self.round >= len(self.potential_experts):
            # current_best_exp_index = np.argmax(list(self.avg_estimated.values()))
            # current_best_exp = self.potential_experts[current_best_exp_index]
            # if self.round % 100 == 0:
            #     print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:f}, disk write: {:f}'.format(self.tot_hoc_hit/self.tot_req_num*100, self.tot_obj_hit/self.tot_req_num*100, self.tot_byte_miss/self.tot_req_bytes*100, self.tot_disk_read, self.tot_disk_write))
            sys.stdout.flush()
            beta = self.calculateBeta()
            z = self.calculateZ()
            self.betas.append(beta)
            self.zs.append(z)
            if z >= beta:
                arm_index = np.argmax(list(self.avg_estimated.values()))
                arm = self.potential_experts[arm_index]
                new_f, new_s = self.extractThres(arm)
                self.freq_thres = new_f
                self.size_thres = new_s
                self.bandit_end = True
                if __debug__:
                    print("Bandit finishes after {:d} requests".format(self.stage_parsed_requests))
        return
        
    def stageTransition(self):
        if self.current_stage == OnlineStage.CACHE_WARMUP and self.stage_parsed_requests == WARMUP_LENGTH:
            assert self.hoc_full
            self.feature_cache.initialize(self.initial_objects, self.obj_sizes, self.initial_times)
            self.current_stage = OnlineStage.FEATURE_COLLECTION
            self.stage_parsed_requests = 0
        if self.current_stage == OnlineStage.FEATURE_COLLECTION and self.checkFeatureConfidence():
            
            self.collectFeature()
            self.cluster()
            self.current_stage = OnlineStage.BEST_DEPLOYED
            self.stage_parsed_requests = 0
        return
    
    # promote an object from dc to hoc
    def promote(self, id, size):
        # delete the object from dc and the dc access table
        self.dc.removeFromCache(id, size)
        del self.dcAccessTab[id]

        disk_write = 0
        
        # when hoc is full, demote the hoc's lru object to the dc
        while self.hoc.current_size + size > self.hoc.cache_size:
            if not self.hoc_full:
                print("HOC full at request {:d}".format(self.stage_parsed_requests))
                # sys.stdout.flush()
            disk_write += self.demote()

        # add the object to hoc
        self.hoc.addToCache(id, size)
        
        return disk_write

    # demote the lru object from hoc to dc
    def demote(self):
        self.hoc_full = True

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
                    disk_write += self.promote(id, size)

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
        if self.tot_req_num % 100000 == 0:
            print('hoc hit: {:.4f}%, disk write: {:.4f}'.format(self.tot_hoc_hit/self.tot_req_num*100, self.tot_disk_write))
            sys.stdout.flush()
        
    def featureCacheInit(self, t, id, size):
        self.obj_reqs[id] += 1
        self.obj_iats[id].append(-1)
        
        if id not in self.obj_sizes:

            self.initial_objects.append(id)        
            self.obj_sizes[id] = size

        self.initial_times[id] = t
        return
    
    def feedFeature(self, t, id, size):
        self.obj_sizes[id] = size
        self.obj_reqs[id] += 1
        self.size_count += 1
        self.size_avg += 1/self.size_count*(size-self.size_avg)

        try:
            k = self.feature_cache.insert(id, size, self.stage_parsed_requests)
        except:
            print("Feature cache insertion error.")

        # TODO: sd for more than 2 occurrences are not exact unique bytes
        sd, iat = k    
        if sd and self.stage_parsed_requests > self.feature_cache.iat_win:
            for num, (s, t) in enumerate(zip(sd, iat)):

                # if s == -1:
                #     break

                self.sd_count[num] += 1
                self.sd_avg[num+1] += 1/self.sd_count[num]*(s-self.sd_avg[num+1])
                self.iat_avg[num+1] += 1/self.sd_count[num]*(t-self.iat_avg[num+1])

        self.obj_iats[id].append(iat)  
    
    def collectFeature(self):
        # edc_count = 0
        # for id in self.feature_cache.items:
        #     n = self.feature_cache.items[id]
        #     edc_count += 1
        #     for i, edc in enumerate(n.edcs):
        #         self.edc_avg[i+1] += 1/edc_count*(edc-self.edc_avg[i+1])
        for features in [self.iat_avg, self.sd_avg, self.size_avg]:
            if type(features) is dict:
                for k, v in sorted(features.items()):
                    self.feature.append(v)
            else:
                self.feature.append(features)
        # print(self.feature)
        # self.feature = [ (self.feature[i]- FEATURE_MIN_LIST[i])/(FEATURE_MAX_LIST[i]-FEATURE_MIN_LIST[i]) for i in range(len(FEATURE_MIN_LIST))]
    
    def feedRequest(self, t, id, size):
        firstTimeReq = True
        if id in self.bloom:
            obj_hit, byte_miss, disk_read, disk_write = self.request(t, id, size)
            firstTimeReq = False
        else:
            obj_hit = disk_read = disk_write = 0
            byte_miss = size
        self.bloom.add(id)
        if self.current_stage == OnlineStage.CACHE_WARMUP:
            if not firstTimeReq:
                self.featureCacheInit(t, id, size)
        else:
            self.collectStat(obj_hit, byte_miss, disk_read, disk_write, size)
        if self.current_stage == OnlineStage.FEATURE_COLLECTION:
            if not firstTimeReq:
                self.feedFeature(t, id, size)
            pass
                
        self.stage_parsed_requests += 1
        self.stageTransition()
        return


def main():
    global model_path
    trace_path, model_path, hoc_s, dc_s = parseInput()
    
    if None not in (trace_path, model_path, hoc_s, dc_s):
        print('trace: {}, HOC size: {}, DC size: {}'.format(trace_path, hoc_s, dc_s))
    else:
        sys.exit(2)
    
    name = trace_path.split('/')[-1].split('.')[0]
    
    global CLUSTER_MODEL_PATH, CLUSTER_RESULT_PATH, FEATURE_MAX_LIST, FEATURE_MIN_LIST
    CLUSTER_MODEL_PATH = os.path.join(model_path, "kmeans_best.pkl")
    CLUSTER_RESULT_PATH = os.path.join(model_path, "cluster_experts_best.pkl")
    rate = int(model_path.split('-')[-1].split('x')[0])
    print("rate: {}".format(rate))
    cache = OnlineHierarchy(name, 3, 50*rate, hoc_s, dc_s, rate)
    
    for line in open(trace_path):
        line = line.split(',')
        t = int(line[0])
        id = int(line[1])
        size = int(line[2])
        cache.feedRequest(t, id, size)
    
    print('hoc hit num: {}, request num: {}'.format(cache.tot_hoc_hit, cache.tot_req_num))
    print('hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk read: {:f}, disk write: {:f}'.format(cache.tot_hoc_hit/cache.tot_req_num*100, cache.tot_obj_hit/cache.tot_req_num*100, cache.tot_byte_miss/cache.tot_req_bytes*100, cache.tot_disk_read, cache.tot_disk_write))
    return 0

if __name__ == '__main__':
    main()
