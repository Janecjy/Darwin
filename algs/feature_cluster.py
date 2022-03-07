import re
from turtle import color
from importlib_metadata import files

from sklearn.cluster import KMeans
from pyclustertend import hopkins
from gap_statistic import OptimalK
from sklearn import metrics
import numpy as np
import math


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import pickle
import time
from collections import defaultdict


thres = 1


def feature_cluster():
    
    # feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'sizes', 'sd_1', 'iat_1', 'sd_2', 'iat_2', 'sd_3', 'iat_3', 'sd_4', 'iat_4', 'sd_5', 'iat_5', 'sd_6', 'iat_6', 'sd_7', 'iat_7']
    feature_set = ['sd_avg', 'iat_avg', 'size_avg', 'edc_avg']
    best_result = pickle.load(open("../cache/output/best_result.pkl", "rb"))
    feature_files = [path for path in os.listdir("../cache/output/features")]
    
    expert_list = []
    for s in [50, 100, 200, 500, 1000]:
        for f in [2, 4, 5, 7]:
            expert_list.append('f'+str(f)+'s'+str(s))
    name_list = []
    feature_list = []
    for file in feature_files:
        feature = []
        name_list.append(file.split('.')[0])
        features = pickle.load(open("../cache/output/features/"+file, "rb"))
        for k, v in features.items():
            if k in feature_set:
                if type(v) is dict or type(v) is defaultdict:
                    values = [value for key,value in sorted(v.items())]   
                    feature += values
                else:
                    feature.append(v)
        # print(len(feature))
        feature_list.append(feature)
    
    print("Clustering with {:d} data points".format(len(feature_list)))
    X = np.array(feature_list)

    # hopkins test for clusterability
    clusterability = hopkins(X, X.shape[0])
    print("Clusterability: %f" % (clusterability))
    assert clusterability < 0.5, "no meaningfull cluster"


    # choose optimal cluster number with gap statistic
    optimalK = OptimalK()
    n_clusters = optimalK(X, cluster_array=np.arange(1, 2*math.sqrt(X.shape[0])))
    print('Optimal clusters: ', n_clusters)

    #     # fig2 = plt.figure()

    #     # plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
    #     # plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
    #     #             optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
    #     # plt.grid(True)
    #     # plt.xlabel('Cluster Count')
    #     # plt.ylabel('Gap Value')
    #     # plt.title('Gap Values by Cluster Count')
    #     # plt.savefig(os.path.join(self.output_path, "seg_{:s}".format(str(self.seg_len))+"_gap.png"))
        
    #     n_clusters = 24

    # build kmeans model
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(X)
    labels = kmeans_model.labels_
        
    pickle.dump(X, open("x.pkl", "wb"))
    pickle.dump(kmeans_model, open("kmeans.pkl", "wb"))

    #     # centroids = kmeans_model.cluster_centers_

    #     # # plot the result points
    #     # fig1 = plt.figure()
    #     # ax = fig1.add_subplot(111, projection='3d')

    #     # x = [i[0] for i in X]
    #     # y = [i[1] for i in X]
    #     # z = [i[2] for i in X]
    #     # c = [i[3] for i in X]

    #     # img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    #     # fig1.colorbar(img,  pad=0.08)
    #     # img2 = ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], c=centroids[:, 3], s = 80, cmap=plt.gray())
    #     # fig1.colorbar(img2,  pad=0.09)
    #     # plt.savefig(os.path.join(self.output_path, "seg_{:s}".format(str(self.seg_len)))+'.png')
    
    # calculate Calinski-Harabasz score
    score = metrics.calinski_harabasz_score(X, labels)
    print("Calinski-Harabasz Score: ", score)

    # check best expert within each cluster
    best_expert_dist = dict.fromkeys(range(n_clusters))
    cluster_count = defaultdict(int) # cluster: num of segments
    for k in best_expert_dist.keys():
        best_expert_dist[k] = defaultdict(int) # expert: probability
    
    for i, lab in enumerate(labels):
        cluster_count[lab] += 1
        for e in best_result[name_list[i]]:
            best_expert_dist[lab][e] += 1
    
    
    cmap = matplotlib.cm.get_cmap("Set3").colors
    cmap += matplotlib.cm.get_cmap("Set2").colors
    plot1 = plt.figure(figsize=(20,5))
    
    count = 0
    for i, k in enumerate(best_expert_dist.keys()):
        
        data = []
        for e in best_expert_dist[k]:
            best_expert_dist[k][e] /= cluster_count[i]
            best_expert_dist[k][e] *= 100
        for e in expert_list:
            if e in best_expert_dist[k]:
                data.append(best_expert_dist[k][e])
            else:
                data.append(0)
        print(len(range(len(data))))
        print(len(data))
        print("cluster"+str(k))
        plt.plot(range(len(data)), data, label="cluster"+str(k))
        count += 1
    
    plt.xlabel("expert")
    plt.xticks(range(len(expert_list)), expert_list)
    plt.ylabel("best expert percentage")
    plt.legend()
    plt.title("Best expert percentage of clusters")
    plt.savefig('fig/clusters.png', bbox_inches='tight')
    print([value for key,value in sorted(cluster_count.items())])
    
    
    # for point, label in zip(X, labels):
    #     best_expert_result[label][np.argmax(point)] += 1
    # for k in best_expert_result.keys():
    #     best_expert[k] = best_expert_result[k].index(max(best_expert_result[k]))
    # for point, label in zip(X, labels):
    #     best_index = best_expert[label]
    #     real_best = np.max(point)
    #     estimate_best = point[best_index]
    #     best_expert_loss[label].append((real_best-estimate_best)/real_best)
    # for k in best_expert_result.keys():
    #     print(f"Cluster {k} has an average best expert bhr loss of {sum(best_expert_loss[k])/len(best_expert_loss[k])*100}% over {len(best_expert_loss[k])} points")

    # return None

def result_cluster():
    best_result = pickle.load(open("../cache/output/best_result.pkl", "rb"))
    expert_list = []
    
    for f in [2, 4, 5, 7]:
        for s in [50, 100, 200, 500, 1000]:
            expert_list.append('f'+str(f)+'s'+str(s))
            
    best_label = []
    cluster_map = {}
    for trace, expert in best_result.items():
        expert_set = []
        for e in expert:
            expert_set.append(expert_list.index(e))
        if expert_set not in best_label:
            best_label.append(expert_set)
        cluster_map[trace] = best_label.index(expert_set)
        
    cluster_count = {}
    for v in cluster_map.values():
        cluster_count[v] = cluster_count.get(v, 0)+1
            
    # print(expert_list)
    # print(best_label)
    # print(cluster_map)
    # print(cluster_count)
    
    feature = 'size_avg'
    feature_files = [path for path in os.listdir("../cache/output/features")]
    cluster1 = []
    cluster5 = []
    cluster6 = []
    cluster10 = []
    for file in feature_files:
        features = pickle.load(open("../cache/output/features/"+file, "rb"))
        if cluster_map[file.split('.')[0]] == 1:
            cluster1.append(features[feature])
            # cluster1.append(features[feature].values())
        if cluster_map[file.split('.')[0]] == 5:
            cluster5.append(features[feature])
            # cluster5.append(features[feature].values())
        if cluster_map[file.split('.')[0]] == 6:
            cluster6.append(features[feature])
            # cluster6.append(features[feature].values())
        if cluster_map[file.split('.')[0]] == 10:
            cluster10.append(features[feature])
            # cluster10.append(features[feature].values())
    
    cmap = matplotlib.cm.get_cmap("Set3").colors

    fig = plt.figure()
    for line in cluster1:
        plt.scatter(1, line, color=cmap[0], marker='o')
        # plt.plot([x+1 for x in range(len(line))], line, color=cmap[0])
    for line in cluster5:
        plt.scatter(2, line, color=cmap[5], marker='^')
        # plt.plot([x+1 for x in range(len(line))], line, color=cmap[5])
    for line in cluster6:
        plt.scatter(3, line, color=cmap[2], marker='x')
        # plt.plot([x+1 for x in range(len(line))], line, color=cmap[2])
    for line in cluster10:
        plt.scatter(4, line, color=cmap[3], marker='+')
        # plt.plot([x+1 for x in range(len(line))], line, color=cmap[3])
    # plt.xticks([x+1 for x in range(len(line))], ['avg_edc'+str(x+1) for x in range(len(line))])
    plt.xticks([x+1 for x in range(4)], ['cluster'+str(x+1) for x in range(4)])
    plt.title(feature)
    plt.savefig('fig/avgsize.png')
        
def countStat(dirPath):   
    hr = {}
    disk_write = {}
    baseline = []
    
    for root, dirs, files in os.walk(dirPath):
        for file in files:

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

    return best_set

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def main():
    # dirs = [path for path in os.listdir("../cache/output") if path.startswith('tc')]
    # best_result = {}
    # for dir in dirs:
    #     best_set = countStat("../cache/output/"+dir)
    #     best_result[dir] = confSort(best_set)
    # pickle.dump(best_result, open("../cache/output/best_result.pkl", "wb"))
    # result_cluster()
    feature_cluster()
    

    
if __name__ == '__main__':
    main()
    