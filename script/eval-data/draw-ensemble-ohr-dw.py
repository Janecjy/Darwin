import random
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, style='white')
ohr_online_result = pickle.load(open("/mydata/results-online/ohr.pkl", "rb"))
dw_online_result = pickle.load(open("/mydata/results-online/dw.pkl", "rb"))

# best map based one single best expert
best_map = {} # best expert: [traces]
for trace in ohr_online_result.keys():
    values = []
    if "online" not in ohr_online_result[trace].keys():
        print(trace)
        continue
    for k in ohr_online_result[trace].keys():
        if k.startswith("f"):
            ohr = ohr_online_result[trace][k]
            dw = dw_online_result[trace][k]
            values.append(ohr-dw/99000000)
    if len(values) > 0:
        max_v = max(values)
    for k in ohr_online_result[trace].keys():
        if ohr_online_result[trace][k]-dw_online_result[trace][k]/99000000 == max_v:
            best = k
    if best not in best_map.keys():
        best_map[best] = []
    best_map[best].append(trace)

trace_list = []
for k in best_map.keys():
    # diff_map = {} # trace: diff
    # for t in best_map[k]:
    #     min_v = min(online_result[t].values())
    #     diff_map[t] = online_result[t]["online"]-min_v
    # trace_list.append(min(diff_map, key=diff_map.get))
    trace_list.append(random.choice(best_map[k]))
print(trace_list)
    
baseline_list = []
# for f in [2, 3, 4, 5, 6, 7]:
for f in [2, 4, 5, 6, 7]:
    # for s in [20, 500]:
    for s in [20, 500]:
        baseline_list.append('f'+str(f)+'s'+str(s))
# baseline_list.append("percentile")
# baseline_list.append("hillclimbing")

diff_data = []
diff_percentage_data = []
for baseline in baseline_list:
    print(baseline)
    diff = []
    diff_percentage = []
    for trace in trace_list:
        if baseline not in ohr_online_result[trace].keys():
            print(trace, baseline)
            continue
        baseline_r = ohr_online_result[trace][baseline]-dw_online_result[trace][baseline]/99000000
        online_r = ohr_online_result[trace]["online"]-dw_online_result[trace]["online"]/99000000
        diff.append(online_r - baseline_r)
        diff_percentage.append((online_r - baseline_r)/baseline_r*100)
        # baseline_data.append(result[trace]['6exp-online']-result[trace][baseline])
    diff_data.append(diff)
    diff_percentage_data.append(diff_percentage)
    print("for baseline {}, the avg reduced is {}, the avg reduced rate is {}".format(baseline, sum(diff)/len(diff), sum(diff_percentage)/len(diff_percentage)))

pickle.dump(diff_data, open("/mydata/results-online/diff_data_ohr_dw.pkl", "wb"))    
pickle.dump(diff_percentage_data, open("/mydata/results-online/diff_percentage_data_ohr_dw.pkl", "wb"))   

sns.set_palette(palette=sns.color_palette("deep"))
diff_data = pickle.load(open("/mydata/results-online/diff_data_ohr_dw.pkl", "rb"))
diff_percentage_data = pickle.load(open("/mydata/results-online/diff_percentage_data_ohr_dw.pkl", "rb"))
diff_data_dict = {}
diff_percentage_data_dict = {}
for i, baseline in enumerate(baseline_list):
    diff_data_dict[baseline] = diff_data[i]
    diff_percentage_data_dict[baseline] = diff_percentage_data[i]
df_diff = pd.DataFrame.from_dict(diff_data_dict)
df_diff_percentage = pd.DataFrame.from_dict(diff_percentage_data_dict)

# sns.set_palette("deep")
plt.figure()
ax = sns.boxplot(data=df_diff, color="C10")
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.axhline(0, color="C3")
plt.xlabel("Improvement (%)")
plt.savefig("ohr-dw-baseline-improvement.png",bbox_inches='tight')

plt.figure()
ax = sns.boxplot(data=df_diff_percentage, color="C10")
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.axhline(0, color="C3")
plt.xlabel("Improvement Rate (%)")
# ax = sns.lineplot(x=[0]*len(baseline_list), y=[y*5 for y in range(len(baseline_list))], ax=ax, linewidth = 2)
plt.savefig("ohr-dw-baseline-improvement-percentage.png",bbox_inches='tight')

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