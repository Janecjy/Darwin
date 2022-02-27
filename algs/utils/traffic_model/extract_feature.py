from cProfile import label
import sys
from parser import *
from collections import defaultdict

from numpy.core import numeric
from gen_trace import *
from treelib import *
from util import *
import random
import pickle
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import dist

TB = 1000000000
MB = 1000000
MIL = 1000000
IAT_GRAN = 200
SD_GRAN = 200000
SIZE_GRAN = 10000

## objects are assumed to be in KB
class cache:
    
    def __init__(self, max_sz):
        self.max_sz = max_sz
        self.items = defaultdict()
        self.curr_sz = 0
        self.debug = open("tmp.txt", "w")
        self.no_del = 0
        
    def initialize(self, initial_objects, sizes, initial_times):        

        ## create a tree structure
        trace_list, self.curr_sz = gen_leaves(initial_objects, sizes, None, self.items, initial_times)
        st_tree, lvl = generate_tree(trace_list)
        root = st_tree[lvl][0]
        root.is_root = True
        self.curr = st_tree[0][0]
        self.prev_rid = root.id        
        

    ## If object is in cache return the appropriate stack distance and inter-arrival-time
    ## Else insert the object at the head and return sd = -1 and iat = -1
    def insert(self, o, sz, tm):
        dts = [] # dt_1, ..., dt_7
        sds = []
        
        if o in self.items:            

            n = self.items[o]
            # dt = tm - n.last_access
            for x in reversed(n.last_access):
                if x > -1:
                    dts.append(tm-x)
                else:
                    dts.append(-1)
            n.last_access.pop(0)
            n.last_access.append(tm)
            
            if self.curr.obj_id == o:
                sd = 0
                sds.append(sd)
                new_sd = sd
                for x in reversed(n.sd_his):
                    if x > -1:
                        new_sd += x
                        sds.append(new_sd)
                    else:
                        sds.append(-1)
                n.sd_his.pop(0)
                n.sd_his.append(sd)
                return sds, dts
            
            sd = self.curr.findUniqBytes(n, self.debug) + self.curr.s + n.s
            sds.append(sd)
            new_sd = sd
            for x in reversed(n.sd_his):
                if x > -1:
                    new_sd += x
                    sds.append(new_sd)
                else:
                    sds.append(-1)  
            n.sd_his.pop(0)
            n.sd_his.append(sd)
            n.delete_node(self.debug)
            self.curr_sz -= n.s           
            n.s = sz            
            # n.last_access = tm
            n.set_b()

            p_c = self.curr.parent            
            self.root = p_c.add_child_first_pos(n, self.debug)
            self.curr_sz += n.s

            if self.root.id != self.prev_rid:
                self.prev_rid = self.root.id
                
            self.curr = n
            
        else:

            n = node(o, sz)
            n.set_b()
            n.last_access = [-1]*(MAX_HIST-1)
            n.last_access.append(tm)
            n.sd_his = [-1]*(MAX_HIST-1)
            
            self.items[o] = n

            p_c = self.curr.parent
            self.root = p_c.add_child_first_pos(n, self.debug)

            if self.root.id != self.prev_rid:
                self.prev_rid = self.root.id
                            
            self.curr = n
            self.curr_sz += sz
                                
            sds = None
            dts = None
            
        ## if cache not full
        while self.curr_sz > self.max_sz:
            try:
                sz, obj = self.root.delete_last_node(self.debug)
                self.curr_sz -= sz
                del self.items[obj]
                self.no_del += 1
            except:
                print("no of deletions : ", self.no_del, obj, o)

        return sds, dts

def main():
    
    print("start")  
    
    name = sys.argv[1]
                  
    lru             = cache(10*TB)
    initial_objects = list()
    initial_times   = {}

    ## Required quantities to be processed later
    obj_sizes         = defaultdict(int)
    obj_iats          = defaultdict(list)
    iats = dict.fromkeys([x+1 for x in range(7)]) # iat distribution, iat[k] is the dt between first arrival and k+1 th arrival
    for x in iats.keys():
        iats[x] = defaultdict(int)
    iat_avg = dict.fromkeys([x+1 for x in range(MAX_HIST)]) # iat average, iat[k] is the dt average between first arrival and k+1 th arrival
    for x in iat_avg.keys():
        iat_avg[x] = 0
    # sd_distances      = defaultdict(list)
    sds = dict.fromkeys([x+1 for x in range(7)]) # sd distribution, sd[k] is the uniqbytes between first arrival and k+1 th arrival
    for x in sds.keys():
        sds[x] = defaultdict(int)
    sd_avg = dict.fromkeys([x+1 for x in range(MAX_HIST)]) # sd average, iat[k] is the dt average between first arrival and k+1 th arrival
    for x in sd_avg.keys():
        sd_avg[x] = 0
    sd_count = [0]*MAX_HIST # sd counts for sd_k for incremental average calculation
    # sd_byte_distances = defaultdict(lambda : defaultdict(lambda : 0))
    sizes = defaultdict(int) # request size distribution
    size_avg = 0
    size_count = 0
    obj_reqs          = defaultdict(int)
    bytes_in_cache    = 0
    line_count        = 0

    # input_file        = sys.argv[1]

    f = open("/home/janechen/cache/traces/feb3/"+name+".txt", "r")

    ## Initialize the LRU stack with objects from the trace
    i = 0
    while bytes_in_cache < 10*MIL:

        l   = f.readline()
        l   = l.strip().split(",")
        tm  = int(l[0])
        obj = int(l[1])
        sz  = int(l[2])*1000
        
        obj_reqs[obj] += 1
        obj_iats[obj].append(-1)
        
        if obj not in obj_sizes:

            initial_objects.append(obj)        
            obj_sizes[obj] = sz
            bytes_in_cache += sz

        initial_times[obj] = tm

        i += 1
        line_count += 1
        if line_count % 100000 == 0:
            print(line_count)

    lru.initialize(initial_objects, obj_sizes, initial_times)

    ## Stats to be processed later
    i          = 0
    line_count = 0
    max_len    = 200000000
    start_tm   = 0
    total_bytes_req = 0
    total_reqs      = 0
    total_misses    = 0
    bytes_miss      = 0
    line_num = []
    iat_avg_converge = dict.fromkeys([x+1 for x in range(7)])
    for x in iat_avg_converge.keys():
        iat_avg_converge[x] = []
    sd_avg_converge = dict.fromkeys([x+1 for x in range(7)])
    for x in sd_avg_converge.keys():
        sd_avg_converge[x] = []
    sizes_avg_converge = []
    sd_goals = dict.fromkeys([x+1 for x in range(7)])
    iat_goals = dict.fromkeys([x+1 for x in range(7)])
    size_g = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["sizes"]
    size_goals = [value for key,value in sorted(size_g.items())]
    for num in range(7):
        sd_g = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["sd_"+str(num+1)]
        sd_goals[num+1] = [value for key,value in sorted(sd_g.items())]
        iat_g = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["iat_"+str(num+1)]
        iat_goals[num+1] = [value for key,value in sorted(iat_g.items())]
    sd_dis = dict.fromkeys([x+1 for x in range(7)])
    for x in sd_dis.keys():
        sd_dis[x] = []
    iat_dis = dict.fromkeys([x+1 for x in range(7)])
    for x in iat_dis.keys():
        iat_dis[x] = []
    size_dis = []
    sd_avg_goals = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["sd_avg"]
    iat_avg_goals = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["iat_avg"]
    size_avg_goal = pickle.load(open("/home/janechen/cache/output/features/"+name+".pkl", "rb"))["size_avg"]
    sd_avg_dis = dict.fromkeys([x+1 for x in range(7)])
    for x in sd_avg_dis.keys():
        sd_avg_dis[x] = []
    iat_avg_dis = dict.fromkeys([x+1 for x in range(7)])
    for x in iat_avg_dis.keys():
        iat_avg_dis[x] = []
    size_avg_dis = []
    

    ## Stack distance is grouped in multiples of 200 MB and inter-arrival time in 200 seconds
    ## Run the trace through LRU cache to obtain the footprint descriptors
    while True:

        l   = f.readline()
        l   = l.strip().split(",")

        try:
            tm  = int(l[0])
            obj = int(l[1])
            sz  = int(l[2])*1000
        except:
            break
            
        if i == 0:
            start_tm = tm 
            
        obj_sizes[obj] = sz
        obj_reqs[obj] += 1
        sz_ = float(sz)/SIZE_GRAN
        sz_  = int(sz_) * SIZE_GRAN
        if sz_ > MB:
            sz_ = MB
        sizes[sz_] += 1
        size_count += 1
        size_avg += 1/size_count*(sz-size_avg)

        total_bytes_req += sz
        total_reqs      += 1

        try:
            k = lru.insert(obj, sz, tm)
        except:
            break

        # TODO: sd for more than 2 occurrences are not exact unique bytes
        sd, iat = k    
        if sd:
            for num, (s, t) in enumerate(zip(sd, iat)):

                if s == -1:
                    break

                sd_count[num] += 1
                sd_avg[num+1] += 1/sd_count[num]*(s-sd_avg[num+1])
                iat_avg[num+1] += 1/sd_count[num]*(t-iat_avg[num+1])
                s  = float(s)/SD_GRAN
                s  = int(s) * SD_GRAN
                # if s > 100*MB:
                #     s = 100*MB
                t = float(t)/IAT_GRAN
                t = int(t) * IAT_GRAN
                # if t > 50000:
                #     t = 50000
                iats[num+1][t] += 1
                sds[num+1][s] += 1    
                # sd_distances[iat].append(sd)
                # sd_byte_distances[iat][sd] += sz
        else:
            total_misses += 1
            bytes_miss   += sz

        obj_iats[obj].append(iat)        
        i += 1
        
        if line_count%100000 == 0:
            print("Processed : ", line_count)    

        if line_count%10000 == 0 and line_count > 0:
            line_num.append(line_count)
            for x in iat_avg_converge.keys():
                iat_avg_converge[x].append(iat_avg[x])
                sd_avg_converge[x].append(sd_avg[x])
            sizes_avg_converge.append(size_avg)
            
            
            size_val = []
            for num in range(7):
                sd_val = []
                iat_val = []
                count = sd_count[num]
                for k in range(0, 100*MB+1, SD_GRAN):
                    if k in sds[num+1].keys():
                        sd_val.append(sds[num+1][k]/count)
                    else:
                        sd_val.append(0)
                sd_dis[num+1].append(dist(sd_val, sd_goals[num+1]))
                for k in range(0, 50000+1, IAT_GRAN):
                    if k in iats[num+1].keys():
                        iat_val.append(iats[num+1][k]/count)
                    else:
                        iat_val.append(0)
                iat_dis[num+1].append(dist(iat_val, iat_goals[num+1]))
                
            for k in range(0, MB+1, SIZE_GRAN):
                if k in sizes.keys():
                    size_val.append(sizes[k]/line_count)         
                else:
                    size_val.append(0)
            size_dis.append(dist(size_val, size_goals))
            
            for num in range(7):
                iat_avg_dis[num+1].append(abs(iat_avg_goals[num+1]-iat_avg[num+1]))
                sd_avg_dis[num+1].append(abs(sd_avg_goals[num+1]-sd_avg[num+1]))
            size_avg_dis.append(abs(size_avg_goal-size_avg))
            
            
        line_count += 1
        if line_count > max_len:
            break

    end_tm = tm        

    features = dict()
    MAX_IAT = [1000000*i for i in range(7)]
    MAX_SD = [TB*i for i in range(7)]

    for num in range(7):
        count = sd_count[num]
        if max(iats[num+1].keys()) > MAX_IAT[num+1]:
            MAX_IAT[num+1] = max(iats[num+1].keys())
        if max(sds[num+1].keys()) > MAX_SD[num+1]:
            MAX_SD[num+1] = max(sds[num+1].keys())
            
        for k in range(0, MAX_SD+1, SD_GRAN):
            if k in sds[num+1].keys():
                sds[num+1][k] /= count
            else:
                sds[num+1][k] = 0
        for k in range(0, MAX_IAT+1, IAT_GRAN):
            if k in iats[num+1].keys():
                iats[num+1][k] /= count
            else:
                iats[num+1][k] = 0
    for k in range(0, MB+1, SIZE_GRAN):
        if k in sizes.keys():
            sizes[k] /= line_count
        else:
            sizes[k] = 0
    
    print(MAX_IAT)
    print(MAX_SD)
    
    features["sd_avg"] = sd_avg
    features["iat_avg"] = iat_avg
    features["size_avg"] = size_avg
    features["sizes"] = sizes
    
    for num in range(7):
        features["sd_"+str(num+1)] = sds[num+1]
        features["iat_"+str(num+1)] = iats[num+1]
    
    pickle.dump(features, open(os.path.join("/home/janechen/cache/output/features/"+name+".pkl"), "wb"))
    
    # create figure and axis objects with subplots()
    cmap = matplotlib.cm.get_cmap("Set3").colors
    plot1 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in iat_avg_converge.keys():
        plt.plot(line_num, iat_avg_converge[x], color =cmap[count], label="iat_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    ticks = range(1000000, 10000000, 1000000)
    plt.xticks(ticks, ticks)

    # set y-axis label
    plt.ylabel("Average iat")
    plt.legend()
    plt.title(name+" average iat timeline")
    plt.savefig('fig/iat.png', bbox_inches='tight')
    
    plot2 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in iat_avg_converge.keys():
        plt.plot(line_num, sd_avg_converge[x], color =cmap[count], label="sd_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Average sd")
    plt.legend()
    plt.title(name+" average sd timeline")
    plt.savefig('fig/sd.png', bbox_inches='tight')
    
    plot3 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    plt.plot(line_num, sizes_avg_converge, color =cmap[count], label="size-avg")
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Average size")
    plt.legend()
    plt.title(name+" average size timeline")
    plt.savefig('fig/size.png', bbox_inches='tight')
    
    plot4 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in iat_dis.keys():
        plt.plot(line_num, iat_dis[x], color =cmap[count], label="iat_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)

    # set y-axis label
    plt.ylabel("Distance between current iat distribution and final iat distribution")
    plt.legend()
    plt.title(name+" iat distribution distance")
    plt.savefig('fig/iat-dis.png', bbox_inches='tight')
    
    plot5 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in sd_dis.keys():
        plt.plot(line_num, sd_dis[x], color =cmap[count], label="sd_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Distance between current sd distribution and final sd distribution")
    plt.legend()
    plt.title(name+" sd distribution distance")
    plt.savefig('fig/sd-dis.png', bbox_inches='tight')
    
    plot6 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    plt.plot(line_num, size_dis, color =cmap[count], label="size-avg")
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Distance between current size distribution and final size distribution")
    plt.legend()
    plt.title(name+" size distribution distance")
    plt.savefig('fig/size-dis.png', bbox_inches='tight')
    
    plot7 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in iat_avg_dis.keys():
        plt.plot(line_num, iat_avg_dis[x], color =cmap[count], label="iat_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis label
    plt.ylabel("Distance between current iat average and final iat average")
    plt.legend()
    plt.title(name+" iat average distance")
    plt.savefig('fig/iat-avg-dis.png', bbox_inches='tight')
    
    plot8 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    for x in sd_avg_dis.keys():
        plt.plot(line_num, sd_avg_dis[x], color =cmap[count], label="sd_"+str(x))
        count += 1
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Distance between current sd average and final sd average")
    plt.legend()
    plt.title(name+" sd average distance")
    plt.savefig('fig/sd-avg-dis.png', bbox_inches='tight')
    
    plot9 = plt.figure(figsize=(20,5))
    # make a plot
    count = 0
    plt.plot(line_num, size_avg_dis, color =cmap[count], label="size-avg-dis")
    # set x-axis label
    plt.xlabel("line count")
    plt.xticks(ticks, ticks)
    # set y-axis labelå
    plt.ylabel("Distance between current size average and final size average")
    plt.legend()
    plt.title(name+" size average distance")
    plt.savefig('fig/size-avg-dis.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
    
        


