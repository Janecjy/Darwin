import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

input_dir = "/mydata/EU-trace/subtraces/"
output_dir = "/mydata/output-eu/trace_stats/"
fig_output_dir = "/mydata/output-eu/figs/"

potential_trace_list = ['do', 'el', 'dw', 'dq', 'eb', 'az']

# collect trace stats
# iterate through all traces in the dir
for trace in os.listdir(input_dir):
    # for each trace, get its popularity distribution and size distribution
    
    # print(trace)
    freq_table = {} # id: count
    sizes = [] # list of sizes of each request
    size_table = {} # id: size, to make the sizes consistent for the same id
    for line in open(input_dir+trace, "r"):
        id = int(line.split()[1])
        if id not in size_table.keys():
            size = int(line.split()[3])/1024
            freq_table[id] = 0
            size_table[id] = size
        else:
            size = size_table[id]
        sizes.append(size)
        freq_table[id] += 1
    
    # save data as pkl
    pickle.dump(sizes, open(output_dir+trace+"-sizes.pkl", "wb"))
    pickle.dump(freq_table, open(output_dir+trace+"-freq.pkl", "wb"))


# draw request size distribution
i = 0
for trace in os.listdir(output_dir):
    # load size pkl data
    if trace.endswith("-sizes.pkl") and trace.split('-')[0].split('.')[1] in potential_trace_list:
        sizes = pickle.load(open(output_dir+trace, "rb"))
        # print(len(sizes))
    else:
        continue
            
    # draw size
    sorted_sizes = np.sort(sizes)
    p = 1. * np.arange(len(sorted_sizes)) / (len(sorted_sizes) - 1)
    print(i, i%10)
    if i % 10 == 0:
        if i > 0:
            plt.xscale("log")
            plt.ylabel("CDF")
            plt.xlabel("Request Sizes (KB)")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim([0, 1])
            print(i/10)
            plt.savefig(fig_output_dir+"req-size-"+str(i/10)+".png")
        fig = plt.figure(figsize=(10, 8))
        
    plt.plot(sorted_sizes, p, label=trace.split('-')[0].split('.')[1])
    i += 1

plt.xscale("log")
plt.ylabel("CDF")
plt.xlabel("Request Sizes (KB)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim([0, 1])
plt.savefig(fig_output_dir+"req-size-"+str(i)+".png")


# draw popularity distribution
i = 0
for trace in os.listdir(output_dir):
    # load size pkl data
    if trace.endswith("-freq.pkl") and trace.split('-')[0].split('.')[1] in potential_trace_list:
        freq_table = pickle.load(open(output_dir+trace, "rb"))
        # print(len(sizes))
    else:
        continue
            
    # draw size
    freqs = list(filter(lambda freq: freq > 1, list(freq_table.values())))
    freqs.sort(reverse=True)
    tot_req = sum(freqs)
    tot_obj = len(freqs)
    x = []
    y = []
    req_count = 0
    for j, f in enumerate(freqs):
        x.append(j/tot_obj*100)
        req_count += f
        y.append(req_count/tot_req*100)
    print(i, i%10)
    if i % 10 == 0:
        if i > 0:
            plt.ylabel("Request Percentage (%)")
            plt.xlabel("Object Percentage (%)")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlim([0, 100])
            plt.ylim([0, 100])
            print(i/10)
            plt.savefig(fig_output_dir+"req-freq-"+str(i/10)+".png")
        fig = plt.figure(figsize=(10, 8))
        
    plt.plot(x, y, label=trace.split('-')[0].split('.')[1])
    i += 1

plt.ylabel("Request Percentage (%)")
plt.xlabel("Object Percentage (%)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim([0, 100])
plt.ylim([0, 100])
plt.savefig(fig_output_dir+"req-freq-"+str(i)+".png")