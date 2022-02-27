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
    disk_write = {}
    baseline = []
    
    for root, dirs, files in os.walk(dirPath):
        for file in files:

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


            if file.endswith("s0.txt"):
                # print(file_res)
                for x in file_res:
                    baseline.append(((x[0], x[3])))
                continue

            if file_res:
                hr['f'+f+'s'+s] = [x[0] for x in file_res]
                disk_write['f'+f+'s'+s] = [x[3] for x in file_res]

    # improvement = {}
    improve_minus = {}
    for x in hr.keys():
        # improvement[x] = 2*(hr[x][0] - baseline[0][0]) / (disk_write[x][0] - baseline[0][0])
        improve_minus[x] = (hr[x][0] - baseline[0][0]) - (disk_write[x][0] - baseline[0][1]) * 1e-7
        # improve_minus[x] = hr[x][0] - disk_write[x][0] * 1e-7

    # hr_max = max(list([hr[x][0] for x in hr.keys()]))
    imp_max = max(list([improve_minus[x] for x in improve_minus.keys()]))
    best_set = []
    for x in hr.keys():
        if imp_max - improve_minus[x] < thres:
            best_set.append(x)
        # if hr_max - hr[x][0] < thres:
        #     best_set.append(x)

    return hr, disk_write, improve_minus, best_set

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def drawSeparate(name):

    dirPath = "/home/janechen/cache/output/"+name

    hr, disk_write, improve_minus, best_set = countStat(dirPath)
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

def drawTog(best_sets, best_confs):
    not_included = list(best_sets.keys())
    X = []
    Y = []
    for best_conf in best_confs:
        x = []
        y = []
        for conf in best_sets.keys():
            if best_conf in best_sets[conf]:
                if conf in not_included:
                    not_included.remove(conf)
                x.append(conf[0]/(conf[0]+conf[1]))
                y.append(conf[1]/(conf[0]+conf[1]))
        X.append(x)
        Y.append(y)
    print(X)
    print(Y)
    index=0
    markers = ["." , "," , "v" , "^" , "<", ">"]
    plt.figure()
    for x, y in zip(X, Y):
        plt.scatter(x, y, s=10, marker=markers[index], color=cm.Dark2(index), label=best_confs[index])
        index += 1
    plt.xlabel("TC-0 Percentage")
    plt.ylabel("TC-1 Percentage")
    plt.title("Best Expert for Tragen TC-0 & 1 Traffic Mix")
    plt.legend()
    plt.savefig("./output.png")

    if not_included:
        print(not_included)


if __name__ == '__main__':
    path = sys.argv[1]
    best_sets = {}
    best_confs = ['f4s50', 'f2s1000', 'f2s50']
    for file in os.listdir(path):
        print(file)
        if file.startswith("tc"):
            best_set = drawSeparate(file)
            mixture = file.split('-')[4].split(':')
            x = int(mixture[0])
            y = int(mixture[1])
            best_sets[(x, y)] = confSort(best_set)
    print(best_sets)
    drawTog(best_sets, best_confs)