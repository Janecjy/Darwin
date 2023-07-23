import pickle
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from pyclustertend import hopkins
from gap_statistic import OptimalK
from sklearn import metrics
import numpy as np
import math
from collections import defaultdict
import sys

feature_dir = sys.argv[1]
output_dir = sys.argv[2]
model_dir = sys.argv[3]
ratio = sys.argv[4]

def feature_cluster(thres):
    
    feature_set = ['iat_avg', 'sd_avg', 'size_avg']
    # best_result = pickle.load(open("../cache/output/best_result.pkl", "rb"))
    best_result = pickle.load(open(output_dir+"coarse_best_result_"+thres+".pkl", "rb"))
    feature_files = [path for path in os.listdir(feature_dir)]
    
    expert_list = []
    for s in [10*ratio, 20*ratio, 50*ratio, 100*ratio, 500*ratio, 1000]:
        for f in [2, 3, 4, 5, 6, 7]:
            expert_list.append('f'+str(f)+'s'+str(s))
    name_list = []
    feature_list = []
    for dir in feature_files:
        if dir.endswith("-7") or dir.endswith("-8") or dir.endswith("-9"):
            continue
        feature = []
        name_list.append(dir.split('/')[-1])
        features = pickle.load(open(os.path.join(dir, "3M.pkl"), "rb"))
        for f in feature_set:
            v = features[f]
            if type(v) is dict or type(v) is defaultdict:
                values = [value for key,value in sorted(v.items())]   
                feature += values
            else:
                feature.append(v)
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
    # n_clusters = 45
    print('Optimal clusters: ', n_clusters)

    kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(X)
    labels = kmeans_model.labels_
    
    # build 
        
    # pickle.dump(X, open("/mydata/results/x.pkl", "wb"))
    pickle.dump(kmeans_model, open(model_dir+"kmeans_"+thres+".pkl", "wb"))
    
    # calculate Calinski-Harabasz score
    score = metrics.calinski_harabasz_score(X, labels)
    print("Calinski-Harabasz Score: ", score)

    # check best expert within each cluster
    best_expert_dist = dict.fromkeys(range(n_clusters))
    cluster_result = dict.fromkeys(range(n_clusters)) # cluster: traces
    cluster_count = defaultdict(int) # cluster: num of segments
    for k in best_expert_dist.keys():
        best_expert_dist[k] = defaultdict(int) # expert: probability
        cluster_result[k] = []
    
    
    for i, lab in enumerate(labels):
        cluster_result[lab].append(name_list[i])
        cluster_count[lab] += 1
        for e in best_result[name_list[i]]:
            best_expert_dist[lab][e] += 1
    # pickle.dump(cluster_result, open("/mydata/results/cluster_result_names.pkl", "wb"))
    
    # best_result = pickle.load(open("../cache/output/best_result.pkl", "rb"))
    
    bestSetDict = {} # cluster num: potential best expert list

    for i in cluster_result.keys():
        best_set = set()
        for trace in cluster_result[i]:
            for exp in best_result[trace]:
                best_set.add(exp)
        bestSetDict[i] = list(best_set)
    pickle.dump(bestSetDict, open(model_dir+"cluster_experts_"+thres+".pkl", "wb"))
    
for thres in ["1", "2", "3", "4", "5"]:
    feature_cluster(thres)