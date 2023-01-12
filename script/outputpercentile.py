import os
import sys
import pickle
from scipy import stats

fs = [2, 3, 4, 5, 6, 7]
ss = [10, 20, 50, 100, 500, 1000]

f_results = {}
s_results = {}

def findPercentile(dirPath, file):
    f_result = {} # f: percentile
    s_result = {} # s: percentile
    
    obj_count = {} # id: count
    size_list = []
    
    for line in open(dirPath+file, "r"):
        id = int(line.split(",")[1])
        size = int(line.split(",")[2])
        if id not in obj_count.keys():
            obj_count[id] = 0
        obj_count[id] += 1
        size_list.append(size)
        
    freq_list = sorted(list(obj_count.values()))
    size_list = sorted(size_list)
    
    for f in fs:
        f_result[f] = stats.percentileofscore(freq_list, f)
        
    for s in ss:
        s_result[s] = stats.percentileofscore(size_list, s)
    
    return f_result, s_result

def main():
    dirPath = "/mydata/traces/"
    for file in os.listdir(dirPath):
        f, s = findPercentile(dirPath, file)
        f_results[file.split(".")[0]] = f
        s_results[file.split(".")[0]] = s
        pickle.dump(f_results, open("/mydata/f_percentiles.pkl", "wb"))
        pickle.dump(s_results, open("/mydata/s_percentiles.pkl", "wb"))

if __name__ == '__main__':
    main()
    