import sys
import os
import pickle
import math
import time
import random
from random import choices
from collections import defaultdict

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

WARMUP_LENGTH = 1000000
TRAINING_LEN = 2000000
PREDICTION_LEN = 2000000
ROUND_LEN = 500000
MB = 1000000
MAX_LIST = [0] * 15
MIN_LIST = [0] * 15

def gen_data(expert_0, expert_1, trace, trace_name):

    feature_set = ['iat_avg', 'sd_avg', 'size_avg']
    
    feature = []
        
    features = pickle.load(open(os.path.join("/mydata/features/", trace_name, "3M.pkl"), "rb"))
    for f in feature_set:
        v = features[f]
        if type(v) is dict or type(v) is defaultdict:
            values = [value for key,value in sorted(v.items())]   
            feature += values
        else:
            feature.append(v)
    
    assert len(feature) == 15
    
    inputs = []
    labels = []
    bucket_list = [10, 20, 50, 100, 500, 1000, 5000]
    
    hit_hit_prob = 0 # pi(e1_hit | e0_hit)
    e0_hit_count = 0
    e0_miss_count = 0
    hit_miss_prob = 0 # pi(e1_hit | e0_miss)
    bucket_count = [0]*(len(bucket_list)+1)
    
    with open(os.path.join("/mydata/output-offline/", trace_name, expert_0+"-hits.txt"), 'r') as e0_file, open(os.path.join("/mydata/output-offline/", trace_name, expert_1+"-hits.txt"), 'r') as e1_file, open(trace, 'r') as trace_file:
        count = 0
        
        for _ in range(WARMUP_LENGTH):
            next(trace_file)
        
        for e0, e1, s in zip(e0_file, e1_file, trace_file):
            e0 = int(e0.strip())  # Remove any leading or trailing whitespace from line1
            e1 = int(e1.strip())  # Remove any leading or trailing whitespace from line2
            s = int(s.strip().split(',')[2])
            if count > 0 and count % ROUND_LEN == 0:
                hit_hit_prob = hit_hit_prob/e0_hit_count
                hit_miss_prob = hit_miss_prob/e0_miss_count
                input = []
                input.extend(feature)
                input.extend(bucket_count)
                inputs.append([input, e0_hit_count, e0_miss_count])
                labels.append([hit_hit_prob, hit_miss_prob])
                e0_hit_count = e0_miss_count = hit_hit_prob = hit_miss_prob = 0
                bucket_count = [0]*(len(bucket_list)+1)
            if e0 == 1:
                e0_hit_count += 1
                if e1 == 1:
                    hit_hit_prob += 1
            else:
                e0_miss_count += 1
                if e1 == 1:
                    hit_miss_prob += 1
                    
            for j in range(len(bucket_list)):
                if s < bucket_list[j]:
                    bucket_count[j] += 1
                    break
            if s >= bucket_list[-1]:
                bucket_count[-1] += 1
            count += 1
        if count > 0 and count % ROUND_LEN == 0:
            hit_hit_prob = hit_hit_prob/e0_hit_count
            hit_miss_prob = hit_miss_prob/e0_miss_count
            input = []
            input.extend(feature)
            input.extend(bucket_count)
            inputs.append([input, e0_hit_count, e0_miss_count])
            labels.append([hit_hit_prob, hit_miss_prob])
    
    pickle.dump(inputs, open(os.path.join("/mydata/correlations/", expert_0+"-"+expert_1, trace_name+"-input.pkl"), "wb"))
    pickle.dump(labels, open(os.path.join("/mydata/correlations/", expert_0+"-"+expert_1, trace_name+"-labels.pkl"), "wb"))



def main():
    expert_0 = sys.argv[1]
    expert_1 = sys.argv[2]
    trace = sys.argv[3]
    trace_name = trace.split('/')[-1].split('.')[0]
        
    if os.path.exists(os.path.join("/mydata/output-offline/", trace_name, expert_0+"-hits.txt")) and os.path.exists(os.path.join("/mydata/output-offline/", trace_name, expert_1+"-hits.txt")):
        gen_data(expert_0, expert_1, trace, trace_name)
    else:
        print("{} or {} not exist".format(os.path.join("/mydata/output-offline/", trace_name, expert_0+"-hits.txt"), os.path.join("/mydata/output-offline/", trace_name, expert_1+"-hits.txt")))
        return

if __name__ == '__main__':
    main()