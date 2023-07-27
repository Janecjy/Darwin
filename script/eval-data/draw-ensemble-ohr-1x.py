import random
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

sns.set(font_scale=3, style='white')
rate = int(sys.argv[1])
BASE_DIR = "/scratch2/09498/janechen/"
online_result = pickle.load(open(BASE_DIR+"tragen-output-online-"+str(rate)+"x/results.pkl", "rb"))
offline_result = pickle.load(open(BASE_DIR+"tragen-output-offline-"+str(rate)+"x/ohr.pkl", "rb"))

# best map based one single best expert
best_map = {} # best expert: [traces]
for trace in online_result.keys():
    values = []
    for k in offline_result[trace].keys():
        values.append(offline_result[trace][k])
    if len(values) > 0:
        max_v = max(values)
    for k in offline_result[trace].keys():
        if offline_result[trace][k] == max_v:
            best = k
    if best not in best_map.keys():
        best_map[best] = []
    best_map[best].append(trace)

trace_list = []
for k in best_map.keys():
    trace_list.append(random.choice(best_map[k]))
print(trace_list)

if rate == 1:
    trace_list = ['tc-0-1-206:59-7', 'tc-0-1-75:190-7', 'tc-0-1-241:24-8', 'tc-0-1-59:206-7', 'tc-0-1-265:0-8', 'tc-0-1-155:110-7', 'tc-0-1-0:265-9', 'tc-0-1-5:260-7', 'tc-0-1-8:257-8', 'tc-0-1-0:265-7']
if rate == 5:
    trace_list = ['tc-0-1-166:99-8', 'tc-0-1-13:252-7', 'tc-0-1-34:230-7', 'tc-0-1-228:37-7', 'tc-0-1-123:142-9', 'tc-0-1-0:265-9', 'tc-0-1-21:244-8', 'tc-0-1-161:104-8', 'tc-0-1-2:263-8', 'tc-0-1-0:265-7', 'tc-0-1-2:263-7']

# 1x best
# trace_list = ['tc-0-1-214:51-8', 'tc-0-1-13:252-8', 'tc-0-1-257:8-8', 'tc-0-1-128:136-7', 'tc-0-1-265:0-8', 'tc-0-1-34:230-7', 'tc-0-1-0:265-9', 'tc-0-1-5:260-8', 'tc-0-1-8:257-7', 'tc-0-1-0:265-7']
# trace_list = list(online_result.keys())
# trace_list = ['tc-0-1-150:115', 'tc-0-1-136:128', 'tc-0-1-80:185', 'tc-0-1-244:21', 'tc-0-1-2:263', 'tc-0-1-206:59', 'tc-0-1-0:265', 'tc-0-1-265:0', 'tc-0-1-8:257']
    
baseline_list = []
for f in [2, 3, 4, 5, 6, 7]:
# # for f in [2, 3, 4]:
    # for s in [20, 500]:
    for s in [10, 20, 50, 100, 500, 1000]:
        baseline_list.append('f'+str(f)+'s'+str(s*rate))
baseline_list.append("tragen-output-percentile-"+str(rate)+"x")
# baseline_list.append("hillclimbing")
baseline_list.append("tragen-output-hillclimbing-c10-"+str(rate)+"x")
baseline_list.append("tragen-output-hillclimbing-c20-"+str(rate)+"x")
baseline_list.append("tragen-output-direct-"+str(rate)+"x")
baseline_list.append("tragen-output-adaptsize-"+str(rate)+"x")
# baseline_list.append("hillclimbing-continuous-c20")
# baseline_list.append("hillclimbing-continuous-c50")
# baseline_list.append("hillclimbing-continuous-c100")

diff_data = []
diff_percentage_data = []
for baseline in baseline_list:
    if baseline.startswith('f'):
        baseline_result = offline_result
    else:
        baseline_result = pickle.load(open(BASE_DIR+baseline+"/results.pkl", "rb"))
    
    # print(baseline)
    diff = []
    diff_percentage = []
    for trace in trace_list:
        if trace not in baseline_result.keys() or trace not in online_result.keys() or not baseline_result[trace].values() or not online_result[trace].values():
            diff.append(0)
            diff_percentage.append(0)
            continue
        # print(online_result[trace], baseline_result[trace])
        if baseline.startswith('f'):
            diff.append(list(online_result[trace].values())[0]-baseline_result[trace][baseline])
            diff_percentage.append((list(online_result[trace].values())[0]-baseline_result[trace][baseline])/list(online_result[trace].values())[0]*100)
        else:
            diff.append(list(online_result[trace].values())[0]-list(baseline_result[trace].values())[0])
            diff_percentage.append((list(online_result[trace].values())[0]-list(baseline_result[trace].values())[0])/list(online_result[trace].values())[0]*100)
        # baseline_data.append(result[trace]['6exp-online']-result[trace][baseline])
    diff_data.append(diff)
    diff_percentage_data.append(diff_percentage)
    print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}, the worst improvement rate is {}".format(baseline, sum(diff)/len(diff), sum(diff_percentage)/len(diff_percentage), min(diff_percentage)))

# pickle.dump(diff_data, open("./diff_data.pkl", "wb"))    
# pickle.dump(diff_percentage_data, open("./diff_percentage_data.pkl", "wb"))   

sns.set_palette(palette=sns.color_palette("deep"))
# diff_data = pickle.load(open("./diff_data.pkl", "rb"))
# diff_percentage_data = pickle.load(open("./diff_percentage_data.pkl", "rb"))
diff_data_dict = {}
diff_percentage_data_dict = {}
for i, baseline in enumerate(baseline_list):
    if baseline == "tragen-output-percentile-"+str(rate)+"x":
        baseline = "P"
    if baseline == "tragen-output-hillclimbing-c10-"+str(rate)+"x":
        baseline = "HC-s10"
    if baseline == "tragen-output-hillclimbing-c20-"+str(rate)+"x":
        baseline = "HC-s20"
    if baseline == "tragen-output-direct-"+str(rate)+"x":
        baseline = "Direct"
    if baseline == "tragen-output-adaptsize-"+str(rate)+"x":
        baseline = "AS"
    diff_data_dict[baseline] = diff_data[i]
    diff_percentage_data_dict[baseline] = diff_percentage_data[i]
# print(len(diff_percentage_data_dict["P"]))
# diff_percentage_data_dict["AS"] = [-4, -1.3, -.1, 3, 12.4, 68.2, .7, 5, -.7]
# print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format("adaptsize", sum(diff)/len(diff), sum(diff_percentage_data_dict["AS"])/len(diff_percentage_data_dict["AS"])))
# diff_percentage_data_dict["DM"] = [-.9, -1.2, 1.5, 1.2, 11, 27, 7, .1, .7]
# print("for baseline {}, the avg improvement is {}, the avg improvement rate is {}".format("directmapping", sum(diff)/len(diff), sum(diff_percentage_data_dict["DM"])/len(diff_percentage_data_dict["DM"])))
    
df_diff = pd.DataFrame.from_dict(diff_data_dict)
df_diff_percentage = pd.DataFrame.from_dict(diff_percentage_data_dict)


plt.figure(figsize=(15, 10))
ax = sns.boxplot(data=df_diff_percentage, color="C10")
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.axhline(0, color="C3")
plt.ylabel("HOC OHR Improvement Rate (%)")
# ax = sns.lineplot(x=[0]*len(baseline_list), y=[y*5 for y in range(len(baseline_list))], ax=ax, linewidth = 2)
plt.savefig("baseline-improvement-percentage-"+str(rate)+"x.png",bbox_inches='tight')
