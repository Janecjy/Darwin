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

def feature_cluster(thres):
    
    feature_set = ['iat_avg', 'sd_avg', 'size_avg']
    best_result = pickle.load(open("/mydata/results/3d_coarse_best_result_"+thres+".pkl", "rb"))
    feature_files = [path for path in os.listdir("/mydata/features/train-set")]
    
    expert_list = []
    for r in [10000, 50000, 100000]:
        for s in [20, 1000]:
            for f in [2, 3, 4, 5, 6, 7]:
                expert_list.append('f'+str(f)+'s'+str(s)+'r'+str(r))
    print(len(expert_list))
    name_list = []
    feature_list = []
    for file in feature_files:
        feature = []
        # use = True
        name = file.split('.')[0]
        # for e in expert_list:
        #     if not e in best_result[name].keys():
        #         use = False
        if name not in best_result.keys():
            continue
        name_list.append(name)
        features = pickle.load(open("/mydata/features/train-set/"+file, "rb"))
        for f in feature_set:
            v = features[f]
            if type(v) is dict or type(v) is defaultdict:
                values = [value for key,value in sorted(v.items())]   
                feature += values
            else:
                feature.append(v)
        feature_list.append(feature)
    feature_dict = {}
    for i, feature in enumerate(feature_list):
        feature_dict[name_list[i]] = feature_list[i]
    
    
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
    pickle.dump(kmeans_model, open("/mydata/results/bmr_kmeans_"+thres+".pkl", "wb"))
    
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
                if exp in expert_list:
                    best_set.add(exp)
        bestSetDict[i] = list(best_set)
        print(i, bestSetDict[i], len(bestSetDict[i]))
    pickle.dump(bestSetDict, open("/mydata/results/bmr_cluster_experts_"+thres+".pkl", "wb"))
    
    exp_num = []
    for i in cluster_result.keys():
        for trace in cluster_result[i]:
            exp_num.append(len(bestSetDict[i]))
    pickle.dump(exp_num, open("/mydata/results/exp-num-3d-"+thres+".pkl", "wb"))
    
for thres in ["1", "2", "3", "4", "5"]:
    feature_cluster(thres)