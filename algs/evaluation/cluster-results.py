import os
import pickle
import numpy as np
from sklearn.cluster import KMeans

# feature_input_dir = "/mydata/features/test-set-real/"
# feature_map = {} # trace: features
# features_list = ['iat_avg', 'sd_avg', 'size_avg']
# result_input_dir = "/mydata/results/"
# best_expert_map = {}

# def confSort(keys):
#     return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

# for feature_f in os.listdir(feature_input_dir):
#     trace = feature_f.split('.')[0]
#     features = []
#     input = pickle.load(open(feature_input_dir+feature_f, "rb"))
#     for feat in features_list:
#         if type(input[feat]) is dict:
#             for k, v in sorted(input[feat].items()):
#                 features.append(v)
#         else:
#             features.append(input[feat])
#     feature_map[trace] = features

# for result_f in os.listdir(result_input_dir):
#     if result_f.startswith("results"):
#         results = pickle.load(open(result_input_dir+result_f, "rb"))
#         for trace in results.keys():
#             best_set = []
#             best_result = max(results[trace].values())
#             for exp in results[trace].keys():
#                 if best_result - results[trace][exp] < 1:
#                     best_set.append(exp)
#             best_expert_map[trace] = confSort(best_set)

# # pickle.dump(feature_map, open("/mydata/results/feature-map.pkl", "wb"))
# pickle.dump(best_expert_map, open("/mydata/results/test-best-map.pkl", "wb"))

# feature_map = pickle.load(open("/mydata/results/feature-map.pkl", "rb"))
# best_expert_map = pickle.load(open("/mydata/results/test-best-map.pkl", "rb"))


# for thres in [1, 2, 3, 4, 5]:
#     CLUSTER_MODEL_PATH = "/mydata/results/kmeans_"+str(thres)+".pkl"
#     CLUSTER_RESULT_PATH = "/mydata/results/cluster_experts_"+str(thres)+".pkl"
#     cluster_model = pickle.load(open(CLUSTER_MODEL_PATH, "rb"))
#     cluster_result = pickle.load(open(CLUSTER_RESULT_PATH, "rb"))

#     expert_list = []
#     for f in [2, 3, 4, 5, 6, 7]:
#         for s in [10, 20, 50, 100, 500, 1000]:
#             expert_list.append('f'+str(f)+'s'+str(s))

#     # assert len(feature_map.keys()) == len(best_expert_map.keys())

#     expert_result = {} # trace: potential experts
#     for trace in feature_map.keys():
#         assert (trace in best_expert_map.keys()), trace
#         potential_experts = []
#         feature = feature_map[trace]
#         X_dist = cluster_model.transform([feature])[0]
#         cluster_index = np.argsort(X_dist)[:1]
#         potential_expert_set = set()
#         for cluster in cluster_index:
#             potential_expert_set.update(cluster_result[cluster])
#         for i in confSort(list(potential_expert_set)):
#             if i in expert_list:
#                 potential_experts.append(i)
#         expert_result[trace] = potential_experts

#     pickle.dump(expert_result, open("/mydata/results/expert-result-"+str(thres)+".pkl", "wb"))

# included_rate = {} # thres: count of traces if one in best set is included in potential
# correct_rate = {} # thres: [percentage of potential experts in best set for each trace]
# reduced_rate = {} # thres: [reduced percentage of experts for each trace]
# exp_num = {} # thres: [length of potential experts]

# expert_list = []
# for f in [2, 3, 4, 5, 6, 7]:
#     for s in [10, 20, 50, 100, 500, 1000]:
#         expert_list.append('f'+str(f)+'s'+str(s))

# for thres in [1, 2, 3, 4, 5]:
#     expert_result = pickle.load(open("/mydata/results/expert-result-"+str(thres)+".pkl", "rb"))
#     trace_num = len(expert_result.keys())
#     assert trace_num == 300
    
#     included_rate[thres] = 0
#     correct_rate[thres] = []
#     reduced_rate[thres] = []
#     exp_num[thres] = []
    
#     for trace in expert_result.keys():
#         potential_exps = expert_result[trace]
#         real_exps = best_expert_map[trace]
#         included = False
#         correct_count = 0
#         for p_e in potential_exps:
#             if p_e in real_exps:
#                 included = True
#                 correct_count += 1
#         if included:
#             included_rate[thres] += 1
#         correct_rate[thres].append(correct_count/len(potential_exps))
#         reduced_rate[thres].append((len(expert_list)-len(potential_exps))/len(expert_list)*100)
#         exp_num[thres].append(len(potential_exps))
#     included_rate[thres] = included_rate[thres]/trace_num*100
#     correct_rate[thres] = sum(correct_rate[thres])/len(correct_rate[thres])*100
#     reduced_rate[thres] = sum(reduced_rate[thres])/len(reduced_rate[thres])
    
# pickle.dump(included_rate, open("/mydata/results/included_rate.pkl", "wb"))
# pickle.dump(correct_rate, open("/mydata/results/correct_rate.pkl", "wb"))
# pickle.dump(reduced_rate, open("/mydata/results/reduced_rate.pkl", "wb"))
# pickle.dump(exp_num, open("/mydata/results/exp_num.pkl", "wb"))

thres = [1, 2, 3, 4, 5]

included_rate = pickle.load(open("/mydata/results/included_rate.pkl", "rb"))
correct_rate = pickle.load(open("/mydata/results/correct_rate.pkl", "rb"))
reduced_rate =  pickle.load(open("/mydata/results/reduced_rate.pkl", "rb"))
exp_num =  pickle.load(open("/mydata/results/exp_num.pkl", "rb"))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(font_scale=1.5, style='white')

# plt.figure(figsize=(10, 10))
# ax = sns.barplot(x = list(included_rate.keys()), y = list(included_rate.values()), color="C10")
# # ax.axvline(0, color="C3")
# plt.xlabel("Cluster Threshold (%)", fontsize=25)
# plt.ylabel("Included Rate (%)", fontsize=25)
# plt.savefig("included_rate.png",bbox_inches='tight')

plt.figure(figsize=(10, 10))
ax = sns.barplot(x = list(correct_rate.keys()), y = list(correct_rate.values()), color="C10")
# ax.axvline(0, color="C3")
plt.xlabel("Cluster Threshold (%)", fontsize=25)
plt.ylabel("Correct Rate (%)", fontsize=25)
plt.savefig("correct_rate.png",bbox_inches='tight')

plt.figure(figsize=(10, 10))
ax = sns.barplot(x = list(reduced_rate.keys()), y = list(reduced_rate.values()), color="C10")
# ax.axvline(0, color="C3")
plt.xlabel("Cluster Threshold (%)", fontsize=25)
plt.ylabel("Reduced Rate (%)", fontsize=25)
plt.savefig("reduced_rate.png",bbox_inches='tight')

print(reduced_rate[2])

plt.figure(figsize=(10, 10))
for t in thres:
    sorted_len = np.sort(exp_num[t])
    p = 1. * np.arange(len(sorted_len)) / (len(sorted_len) - 1)
    plt.plot(sorted_len, p, label=str(t)+'%', linewidth=4)
    plt.ylabel("CDF", fontsize=25)
    plt.xlabel("Remaining Expert Number", fontsize=25)
    # plt.legend(bbox_to_anchor=(1.02, 1))
    plt.legend()
    plt.xlim([0, 37])
    plt.ylim([0, 1])
    plt.savefig("exp-num.png",bbox_inches='tight')
    if t == 2:
        for i, cdf in enumerate(p):
            if cdf > 0.9:
                print(sorted_len[i])
        # print(sorted_len, p)
        
    