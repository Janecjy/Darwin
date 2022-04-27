import os
import re
import sys
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

thres = 1

def countStat(dirPath):   
    # print(dirPath) 
    hr = {}
    # disk_write = {}
    # baseline = []
    
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if file.endswith(".txt"):

            # print(file)

                file_res = []

                for line in open(os.path.join(root, file), "r"):
                    val1 = re.findall('freq: [\d]*, size: [\d]*',line)

                    for sentence in val1:
                        sentence = sentence.split(',')
                        f = (sentence[0].split(':')[1].replace(" ", ""))
                        s = (sentence[1].split(':')[1].replace(" ", ""))
                    
                    val2 = re.findall('hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                    
                    for sentence in val2:
                        exp = sentence.split(',')
                        exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                        file_res.append(exp)


                # if file.endswith("s0.txt"):
                #     # print(file_res)
                #     for x in file_res:
                #         baseline.append(((x[0], x[3])))
                #     continue

                if file_res and s != '0':
                    hr['f'+f+'s'+s] = [x[0] for x in file_res]
                    # disk_write['f'+f+'s'+s] = [x[3] for x in file_res]

    # improvement = {}
    # improve_minus = {}
    # for x in hr.keys():
    #     # improvement[x] = 2*(hr[x][0] - baseline[0][0]) / (disk_write[x][0] - baseline[0][0])
    #     improve_minus[x] = (hr[x][0] - baseline[0][0]) - (disk_write[x][0] - baseline[0][1]) * 1e-7
        # improve_minus[x] = hr[x][0] - disk_write[x][0] * 1e-7

    hr_max = max(list([hr[x][0] for x in hr.keys()]))
    # imp_max = max(list([improve_minus[x] for x in improve_minus.keys()]))
    best_set = []
    for x in hr.keys():
        # if imp_max - improve_minus[x] < thres:
        # if (imp_max - improve_minus[x])/imp_max < thres/100:
            # best_set.append(x)
        if (hr_max - hr[x][0])/hr_max < thres/100:
            best_set.append(x)

    # return hr, disk_write, improve_minus, best_set
    return hr, best_set

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def drawSeparate(name):

    dirPath = "/home/janechen/cache/output/test-set-real/"+name

    hr, best_set = countStat(dirPath)
    # print(improve_minus)

    # # selectedConf = [ x for x in confSort(hr.keys()) if not x.startswith("f3") ]
    selectedConf = [ x for x in confSort(hr.keys()) ]
    # # for x in selectedConf:
    # #     print((x, hr[x]))

    # #plot hr
    plt.style.use('ggplot')
    plt.tight_layout()
    plt.figure(figsize=(20,5))

    plt.bar(list(x for x in range(len(selectedConf))), list([hr[x][0] for x in selectedConf]), align='center')
    plt.xticks(list(x for x in range(len(selectedConf))), list(selectedConf), rotation=90)

    plt.xlabel("Threshold Configuration")
    plt.ylabel("Aggregated Hit Rates")
    plt.title(name+" Aggregated Hit Rates", pad=20)

    plt.savefig("fig/"+name+"-hr.png", bbox_inches='tight')

    # # plot disk writes

    # plt.figure(figsize=(20,5))
    # plt.bar(list(x for x in range(len(selectedConf))), list([disk_write[x][0] for x in selectedConf]), align='center')
    # plt.xticks(list(x for x in range(len(selectedConf))), list(selectedConf), rotation=90)

    # plt.xlabel("Threshold Configuration")
    # plt.ylabel("Disk Writes")
    # plt.title(name+" Disk Writes", pad=20)

    # # plt.xticks(x_pos, x)
    # # plt.ylim([1750000, 4000000])

    # plt.savefig("fig/"+name+"-write.png", bbox_inches='tight')

    # # plot disk writes
    # plt.figure(figsize=(20,5))

    # plt.bar(list(x for x in range(len(selectedConf))), list([improve_minus[x] for x in selectedConf]), align='center')
    # plt.xticks(list(x for x in range(len(selectedConf))), list(selectedConf), rotation=90)

    # plt.xlabel("Threshold Configuration")
    # plt.ylabel("HR - Writes * 1e-7")
    # plt.title(name+" AHR", pad=20)

    # plt.savefig("fig/"+name+"-imp.png", bbox_inches='tight')
    return best_set

def drawTog(best_sets, best_resultset):
    
    expert_list = []
    
    for f in [2, 4, 5, 7]:
        for s in [50, 100, 200, 500, 1000]:
            expert_list.append('f'+str(f)+'s'+str(s))
    
    not_included = dict.fromkeys(expert_list, 0) # expert: not included num
    
    X = []
    Y = []
    for best_set in best_resultset:
        x = []
        y = []
        for conf in best_sets.keys():
            if best_sets[conf] == best_set:
                # if conf in not_included:
                #     not_included.remove(conf)
                x.append(conf[0])
                y.append(conf[1])
        X.append(x)
        Y.append(y)
        for e in expert_list:
            if e not in best_set:
                if e == 'f4s50':
                    print(x, y)
                not_included[e] += len(x)
    # print(len(X))
    # print(len(Y))
    # print(len(best_resultset))
    # print(X)
    # print(Y)
    index=0
    # markers = ["." , "," , "v" , "^" , "<", ">"]
    plt.figure()
    for x, y in zip(X, Y):
        plt.scatter(x, y, s=1, color=cm.tab20(index), label=index)
        # plt.scatter(x, y, s=10, marker=markers[index], color=cm.Dark2(index), label=best_resultset[index])
        index += 1
    plt.xlabel("TC-0 Request Rate (req/s)")
    plt.ylabel("TC-1 Request Rate (req/s)")
    plt.title("Best Expert Set for Tragen TC-0 & 1 Traffic Mix (Worse Thres="+str(thres)+'%)')
    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    print(best_resultset)
    print(not_included)
    print(len(best_sets))
    plt.savefig("./output.png")
    
    plt.figure(figsize=(20,5))
    plt.bar(range(len(not_included)), list(not_included.values()), align='center')
    plt.xticks(range(len(not_included)), list(not_included.keys()))
    # # for python 2.x:
    # plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
    # plt.xticks(range(len(D)), D.keys())  # in python 2.x
    # plt.show()
    plt.xlabel("Expert")
    plt.ylabel("Num of Segments Perform Poorly with the Expert")
    plt.title("Expert Not Among Best Expert Set Distribution (Worse Thres="+str(thres)+'%)')
    plt.tight_layout()
    plt.savefig("./output1.png")

    # if not_included:
    #     print(not_included)


def drawSyntheticTraces():
    PATH = "../cache/output/features"
    for file in os.listdir(PATH):
        mixture = file.split('.')[0].split('-')[4].split(':')
        x = int(mixture[0])
        y = int(mixture[1])
        plt.scatter(x, y, s=1)
        plt.savefig("./output.png")

if __name__ == '__main__':
    path = sys.argv[1]
    best_sets = {}
    best_resultset = set()
    for file in os.listdir(path):
        # print(file)
        if file.startswith("tc"):
            _, best_set = countStat(os.path.join(path, file))
            # print(best_set)
            mixture = file.split('-')[4].split(':')
            x = int(mixture[0])
            y = int(mixture[1])
            best_sets[(x, y)] = tuple(confSort(best_set))
            best_resultset.add(tuple(confSort(best_set)))
    # print(best_sets)
    drawTog(best_sets, list(best_resultset))
    # drawSeparate("tc-0-tc-1-2226:676")
    # drawSyntheticTraces()