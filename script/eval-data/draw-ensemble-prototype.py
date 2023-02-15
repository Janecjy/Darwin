import random
import pickle
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, style='white')
online_result = {
    1: {
        "f2s100": 17.328,
        "f2s10": 13.03,
        "f3s100": 18.62,
        "f3s10": 11.78,
        "f2s1000": 15.49,
        "f3s1000": 16.9,
        "f4s1000": 17.52,
        "online": 18.70
    },
    2: {
        "f2s100": 17.099,
        "f2s10": 13.247,
        "f3s100": 5.1585,
        "f3s10": 11.68,
        "f2s1000": 15.74,
        "f3s1000": 15.65,
        "f4s1000": 18.73,
        "online": 18.82
    },
    3: {
        "f2s100": 8.76,
        "f2s10": 18.42,
        "f3s100": 7.315,
        "f3s10": 15.86,
        "f2s1000": 7.29,
        "f3s1000": 6.1,
        "f4s1000": 5.49,
        "online": 20.74
    },
    # 4: {
    #     "f2s100": 69.89,
    #     "f2s10": 40.17,
    #     "f3s100": 5.1585,
    #     "f3s10": 11.68,
    #     "f2s1000": 74.62,
    #     "f3s1000": 61.47,
    #     "f4s1000": 67.75,
    #     "online": 75.59
    # }
}

    
baseline_list = ["f2s10", "f2s100", "f2s1000", "f3s10", "f3s100", "f3s1000", "f4s1000"] # 

diff_data = []
diff_percentage_data = []
for baseline in baseline_list:
    print(baseline)
    diff = []
    diff_percentage = []
    for trace in range(3):
        trace = trace+1
        if baseline not in online_result[trace].keys():
            print(trace, baseline)
            continue
        
        diff.append(online_result[trace]['online']-online_result[trace][baseline])
        diff_percentage.append((online_result[trace]['online']-online_result[trace][baseline])/online_result[trace][baseline]*100)
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
    if baseline == "hillclimbing-continuous-c1":
        baseline = "hillclimbing-s1"
    if baseline == "hillclimbing-continuous-c10":
        baseline = "hillclimbing-s10"
    diff_data_dict[baseline] = diff_data[i]
    diff_percentage_data_dict[baseline] = diff_percentage_data[i]
df_diff = pd.DataFrame.from_dict(diff_data_dict)
df_diff_percentage = pd.DataFrame.from_dict(diff_percentage_data_dict)

# sns.set_palette("deep")
plt.figure()
ax = sns.boxplot(data=df_diff, color="C10")
ax.set_xticklabels(ax.get_xticklabels(),rotation=70)
ax.axhline(0, color="C3")
plt.xlabel("HOC OHR Improvement (%)")
plt.savefig("baseline-improvement.png",bbox_inches='tight')

plt.figure()
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