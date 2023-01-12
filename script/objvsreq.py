import os
import matplotlib.pyplot as plt
import numpy as np
import pickle



def drawPop(file):
    # obj_count = {} # id: req counts
    # for i, line in enumerate(open(file)):
    #     try:
    #         id = int(line.split(',')[1])
    #         if id not in obj_count.keys():
    #             obj_count[id] = 0
    #         obj_count[id] += 1
    #     except:
    #         print(i, line)
    # tot_obj = len(obj_count.keys())
    # tot_req = sum(obj_count.values())
    # print(tot_obj, tot_req)
    # obj_dict = dict(sorted(obj_count.items(), key=lambda item: item[1], reverse=True))
    # pickle.dump(obj_dict, open("obj_dict.pkl", "wb"))

    obj_dict = pickle.load(open("obj_dict.pkl", "rb"))
    
    tot_obj = len(obj_dict.keys())
    tot_req = sum(obj_dict.values())
    print(tot_obj, tot_req)
    
    x = [0]
    y = [0]
    tot_x = [0]
    tot_y = [0]
    obj = 0
    req = 0
    # id_set = set()
    # for line in open("/mydata/traces/"+file):
    #     id = int(line.split(',')[1])
    #     if id not in id_set:
    #         obj += 1
    #         id_set.add(id)
    #     req += 1
    for k, v in obj_dict.items():
        obj += 1
        req += v
        tot_x.append(obj/tot_obj*100)
        tot_y.append(req/tot_req*100)
        if obj > tot_obj*0.1*len(x):
            x.append(obj/tot_obj*100)
            y.append(req/tot_req*100)
    x.append(obj/tot_obj*100)
    y.append(req/tot_req*100)
            
    fig = plt.figure()
    plt.plot(tot_x, tot_y)
    plt.scatter(x, y)
    print(x, y)
    plt.xlabel("Object Percentage (%)")
    plt.ylabel("Request Percentage (%)")
    plt.savefig("popularity-dis.png")

if __name__ == "__main__":
    drawPop("/mydata/jedi-traces/tc1.txt")