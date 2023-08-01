import random
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=3, style='white')
online_result = pickle.load(open("/Users/janechen/Downloads/darwin-data/ohr-results.pkl", "rb"))

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

trace_list = ['tc-0-1-150:115', 'tc-0-1-136:128', 'tc-0-1-80:185', 'tc-0-1-244:21', 'tc-0-1-2:263', 'tc-0-1-206:59', 'tc-0-1-0:265', 'tc-0-1-265:0', 'tc-0-1-8:257']
    
baseline_list = []
for f in [2, 3, 4, 5, 6, 7]:
# for f in [2, 3, 4]:
    # for s in [20, 500]:
    for s in [10, 20, 50, 100, 500, 1000]:
        baseline_list.append('f'+str(f)+'s'+str(s))
baseline_list.append("percentile")
# baseline_list.append("hillclimbing")
baseline_list.append("hillclimbing-continuous-c1")
baseline_list.append("hillclimbing-continuous-c10")
baseline_list.append("hillclimbing-continuous-c20")
baseline_list.append("hillclimbing-continuous-c50")
baseline_list.append("hillclimbing-continuous-c100")

diff_data = []
diff_percentage_data = []
for baseline in baseline_list:
    # print(baseline)
    diff = []
    diff_percentage = []
    for trace in trace_list:
        if baseline not in online_result[trace].keys():
            # print(trace, baseline)
            continue
        
        diff.append(online_result[trace]['online-new']-online_result[trace][baseline])
        diff_percentage.append((online_result[trace]['online-new']-online_result[trace][baseline])/online_result[trace][baseline]*100)
        # baseline_data.append(result[trace]['6exp-online']-result[trace][baseline])
    diff_data.append(diff)
    diff_percentage_data.append(diff_percentage)
    print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format(baseline, sum(diff)/len(diff), sum(diff_percentage)/len(diff_percentage)))

# pickle.dump(diff_data, open("./diff_data.pkl", "wb"))    
# pickle.dump(diff_percentage_data, open("./diff_percentage_data.pkl", "wb"))   

sns.set_palette(palette=sns.color_palette("deep"))
# diff_data = pickle.load(open("./diff_data.pkl", "rb"))
# diff_percentage_data = pickle.load(open("./diff_percentage_data.pkl", "rb"))
diff_data_dict = {}
diff_percentage_data_dict = {}
for i, baseline in enumerate(baseline_list):
    if baseline == "percentile":
        baseline = "P"
    if baseline == "hillclimbing-continuous-c1":
        baseline = "HC-s1"
    if baseline == "hillclimbing-continuous-c10":
        baseline = "HC-s10"
    if baseline == "hillclimbing-continuous-c50":
        baseline = "HC-s50"
    if baseline == "hillclimbing-continuous-c20":
        baseline = "HC-s20"
    if baseline == "hillclimbing-continuous-c100":
        baseline = "HC-s100"
    diff_data_dict[baseline] = diff_data[i]
    diff_percentage_data_dict[baseline] = diff_percentage_data[i]
# print(len(diff_percentage_data_dict["P"]))
diff_percentage_data_dict["AS"] = [-4, -1.3, -.1, 3, 12.4, 68.2, .7, 5, -.7]
print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format("adaptsize", sum(diff)/len(diff), sum(diff_percentage_data_dict["AS"])/len(diff_percentage_data_dict["AS"])))
diff_percentage_data_dict["DM"] = [-.9, -1.2, 1.5, 1.2, 11, 27, 7, .1, .7]
print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format("directmapping", sum(diff)/len(diff), sum(diff_percentage_data_dict["DM"])/len(diff_percentage_data_dict["DM"])))
    
df_diff = pd.DataFrame.from_dict(diff_data_dict)
df_diff_percentage = pd.DataFrame.from_dict(diff_percentage_data_dict)

# sns.set_palette("deep")
# plt.figure(figsize=(12, 10))
# ax = sns.boxplot(data=df_diff, color="C10")
# ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
# ax.axhline(0, color="C3")
# plt.ylabel("HOC OHR Improvement (%)")
# plt.savefig("baseline-improvement.png",bbox_inches='tight')

plt.figure(figsize=(15, 10))
ax = sns.boxplot(data=df_diff_percentage, color="C10")
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.axhline(0, color="C3")
plt.ylabel("HOC OHR Improvement Rate (%)")
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