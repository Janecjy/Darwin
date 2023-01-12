import os
import matplotlib.pyplot as plt
import numpy as np
import pickle


def countOneHitWonders(file):
    # obj_count = {} # id: req counts
    # for line in open("/mydata/traces/"+file):
    #     id = int(line.split(',')[1])
    #     if id not in obj_count.keys():
    #         obj_count[id] = 0
    #     obj_count[id] += 1
    
    # data = []
    # for k, v in obj_count.items():
    #     for i in range(v):
    #         data.append(v)
            
    # sorted = np.sort(data)
        
    # pickle.dump(sorted, open("freq-cdf.pkl", "wb"))
    sorted = pickle.load(open("freq-cdf.pkl", "rb"))
    p = 1. * np.arange(len(sorted)) / (len(sorted) - 1)
    for i, d in enumerate(sorted):
        if d >1:
            print(p[i])
            break
    fig = plt.figure()
    plt.plot(sorted, p)
    plt.xlim([0, 1000])
    plt.savefig("output.png")
    
def countBytes(req_threshold):
    pickle.load(open())

if __name__ == "__main__":
    countOneHitWonders("tc-0-1-0:265.txt")