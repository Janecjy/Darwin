import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def calWorkingSet(trace, conf_set):

    sizeTab = {} # obj: size
    reqCountTab = defaultdict(int) # obj: request num
    confSet = dict.fromkeys(conf_set) # (freq_thres, size_thres): [obj num, tot_obj_size, req num]

    for conf in conf_set:
        confSet[conf] = [0, 0, 0]

    with open(trace, "r") as f:
        for line in f:
            id = int(line.split(',')[1])
            if id not in sizeTab:
                size = int(line.split(',')[2])
                sizeTab[id] = size
            reqCountTab[id] += 1

    for obj in reqCountTab.keys():
        for conf in conf_set:
            if sizeTab[obj] < conf[1] and reqCountTab[obj] >= conf[0]:
                # meet the admission requirements
                confSet[conf][0] += 1
                confSet[conf][1] += sizeTab[obj]
                confSet[conf][2] += reqCountTab[obj]
    
    compulsory_miss = len(sizeTab.keys()) # total num of objs

    return confSet, compulsory_miss

import os
import re

def countStat(dirPath):   
    hr = {}
    
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if not file.endswith("s0.txt"):

                file_res = []

                for line in open(os.path.join(root, file), "r"):
                    val1 = re.findall('freq: [\d]*, size: [\d]*',line)

                    for sentence in val1:
                        sentence = sentence.split(',')
                        f = int((sentence[0].split(':')[1].replace(" ", "")))
                        s = int((sentence[1].split(':')[1].replace(" ", "")))
                    
                    val2 = re.findall('hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                    
                    for sentence in val2:
                        exp = sentence.split(',')
                        exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                        file_res.append(exp)

                if file_res:
                    hr[(f, s)] = [x[0] for x in file_res]

    return hr

name = "tc-1-24300"
conf_set = []
for f in [2, 4, 5, 7]:
    for s in [50, 100, 200, 500, 1000]:
        conf_set.append((f, s))



confSet, compulsory_miss = calWorkingSet("../cache/traces/feb3/"+name+".txt", conf_set)
print(confSet)
print(compulsory_miss)
hr = countStat("../cache/output/"+name)



# create figure and axis objects with subplots()
fig,ax = plt.subplots(figsize=(20,5))
# make a plot
workingset_cachesize_rate = [confSet[conf][1]/100000 for conf in conf_set]
ax.plot(range(len(conf_set)), workingset_cachesize_rate, color="red", marker="o")
# set x-axis label
ax.set_xlabel("Conf")
ax.set_xticks(range(len(conf_set)))
ax.set_xticklabels(conf_set)
# set y-axis label
ax.set_ylabel("Working Set Size / HOC Size",color="red",fontsize=14)
ax.set_title(name+" Working Set Size")

hr_list = [hr[conf] for conf in conf_set]
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(range(len(conf_set)), hr_list,color="blue",marker="o")
ax2.set_ylabel("Hit Rate",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('fig/'+name+'-workingsetsize.png', bbox_inches='tight')



# create figure and axis objects with subplots()
fig,ax = plt.subplots(figsize=(20,5))
# make a plot
workingsetreq_totreq_rate = [confSet[conf][2]/10000000 for conf in conf_set]
ax.plot(range(len(conf_set)), workingsetreq_totreq_rate, color="red", marker="o")
# set x-axis label
ax.set_xlabel("Conf")
ax.set_xticks(range(len(conf_set)))
ax.set_xticklabels(conf_set)
# set y-axis label
ax.set_ylabel("Working Set Req Num / Tot Req Num",color="red",fontsize=14)
ax.set_title(name+" Working Set Req Num")

hr_list = [hr[conf] for conf in conf_set]
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(range(len(conf_set)), hr_list,color="blue",marker="o")
ax2.set_ylabel("Hit Rate",color="blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('fig/'+name+'-workingsetreq.png', bbox_inches='tight')