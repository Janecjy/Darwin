import random
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, style='white')
# online_result = pickle.load(open("/Users/janechen/Downloads/darwin-data/ohr-results.pkl", "rb"))

# # best map based one single best expert
# best_map = {} # best expert: [traces]
# for trace in online_result.keys():
#     values = []
#     for k in online_result[trace].keys():
#         if k.startswith("f"):
#             values.append(online_result[trace][k])
#     if len(values) > 0:
#         max_v = max(values)
#     for k in online_result[trace].keys():
#         if online_result[trace][k] == max_v:
#             best = k
#     if best not in best_map.keys():
#         best_map[best] = []
#     best_map[best].append(trace)

# trace_list = []
# for k in best_map.keys():
#     trace_list.append(random.choice(best_map[k]))
# print(trace_list)
    
baseline_list = []
for f in [2, 3, 4, 5, 6, 7]:
# for f in [2, 3, 4]:
    for s in [10, 20, 50, 100, 500, 1000]:
        baseline_list.append('f'+str(f)+'s'+str(s))
baseline_list.append("percentile")
baseline_list.append("hillclimbing")

# diff_data = []
# diff_percentage_data = []
# for baseline in baseline_list:
#     print(baseline)
#     diff = []
#     diff_percentage = []
#     for trace in trace_list:
#         if baseline not in online_result[trace].keys():
#             print(trace, baseline)
#             continue
        
#         diff.append(online_result[trace]['online-new']-online_result[trace][baseline])
#         diff_percentage.append((online_result[trace]['online-new']-online_result[trace][baseline])/online_result[trace][baseline]*100)
#         # baseline_data.append(result[trace]['6exp-online']-result[trace][baseline])
#     diff_data.append(diff)
#     diff_percentage_data.append(diff_percentage)
#     print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format(baseline, sum(diff)/len(diff), sum(diff_percentage)/len(diff_percentage)))

# pickle.dump(diff_data, open("./diff_data.pkl", "wb"))    
# pickle.dump(diff_percentage_data, open("./diff_percentage_data.pkl", "wb"))   

sns.set_palette(palette=sns.color_palette("deep"))
diff_data = pickle.load(open("./diff_data.pkl", "rb"))
diff_percentage_data = pickle.load(open("./diff_percentage_data.pkl", "rb"))
diff_data_dict = {}
diff_percentage_data_dict = {}
for i, baseline in enumerate(baseline_list):
    diff_data_dict[baseline] = diff_data[i]
    diff_percentage_data_dict[baseline] = diff_percentage_data[i]
df_diff = pd.DataFrame.from_dict(diff_data_dict)
df_diff_percentage = pd.DataFrame.from_dict(diff_percentage_data_dict)

# sns.set_palette("deep")
plt.figure(figsize=(30, 10))
ax = sns.boxplot(data=df_diff, color="C10", orient="h")
ax.axvline(0, color="C3")
plt.xlabel("HOC OHR Improvement (%)")
plt.savefig("baseline-improvement.png",bbox_inches='tight')

plt.figure(figsize=(30, 10))
ax = sns.boxplot(data=df_diff_percentage, color="C10", orient="h")
ax.axvline(0, color="C3")
plt.xlabel("HOC OHR Improvement Rate (%)")
# ax = sns.lineplot(x=[0]*len(baseline_list), y=[y*5 for y in range(len(baseline_list))], ax=ax, linewidth = 2)
plt.savefig("baseline-improvement-percentage.png",bbox_inches='tight')

# # plot improvement
# fig = plt.figure(figsize=(30, 10))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
# ax.grid()
 
# # Creating plot
# bp = ax.boxplot(diff_data, flierprops={'marker': 'o', 'markersize': 3}, meanline=True)
# # ax.set_xticklabels(baseline_list)
# # ax.set_xlabel("Baseline")
# # ax.set_ylabel("Object HOC Hit Rate Improvement (%)")
# plt.xlabel("Baseline")
# plt.ylabel("Object HOC Hit Rate Improvement (%)")
# plt.xticks(range(len(baseline_list)), baseline_list)
# # plt.show()
# plt.savefig("baseline-improvement.png")

# # plot improvement rate
# fig = plt.figure(figsize=(30, 10))
 
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
# ax.grid()
 
# # Creating plot
# bp = ax.boxplot(diff_percentage_data, flierprops={'marker': 'o', 'markersize': 3}, meanline=True)
# # ax.set_xticklabels(baseline_list)
# # ax.set_xlabel("Baseline")
# # ax.set_ylabel("Object HOC Hit Rate Improvement Rate(%)")
# plt.xlabel("Baseline")
# plt.ylabel("Object HOC Hit Rate Improvement Rate (%)")
# plt.xticks(range(len(baseline_list)), baseline_list)
# # plt.show()
# plt.savefig("baseline-improvement-rate.png")