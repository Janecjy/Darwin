import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=2, style='white')

trace_length = 100000
proxy_lat = 20
origin_lat = 100
sigma = 1
hr_result = {
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
    4: {
        "f2s100": 69.89,
        "f2s10": 40.17,
        "f3s100": 5.1585,
        "f3s10": 11.68,
        "f2s1000": 74.62,
        "f3s1000": 61.47,
        "f4s1000": 67.75,
        "online": 75.59
    }
}

exp_list = ["f2s10", "f2s100", "f2s1000", "f3s10", "f3s100", "f3s1000", "f4s1000", "online"]
fb_latency = dict.fromkeys(exp_list)
# e2e_latency = dict.fromkeys(exp_list)
for exp in exp_list:
    fb_latency[exp] = []
    # e2e_latency[exp] = []

for exp in exp_list:
    for trace in hr_result.keys():
    #     for i in range(trace_length):
        hit_num = round(trace_length*hr_result[trace][exp]/100)
        miss_num = trace_length - hit_num
        print(proxy_lat, sigma, hit_num, miss_num)
        hit_array = np.random.normal(proxy_lat, sigma, hit_num)
        miss_array = np.random.normal(origin_lat, sigma, miss_num)
        # print(fb_latency[exp].shape, hit_array.shape, miss_array.shape)
        fb_latency[exp] += list(hit_array)
        fb_latency[exp] += list(miss_array)
        # np.concatenate(fb_latency[exp], hit_array, miss_array)
    assert len(fb_latency[exp]) == trace_length*len(hr_result.keys())

pickle.dump(fb_latency, open("/mydata/fb-latency.pkl", "wb"))

# fb_latency = pickle.load(open("/mydata/fb-latency.pkl", "rb"))

plt.figure(figsize=(12, 10))
for exp in exp_list:
    
    sorted_len = np.sort(fb_latency[exp])
    p = 1. * np.arange(len(fb_latency[exp])) / (len(fb_latency[exp]) - 1)
    if exp == "online":
        exp = "Darwin"
    if exp == "Darwin":
        plt.plot(sorted_len, p, label=exp, linewidth=4, color="tab:cyan")
    else:
        plt.plot(sorted_len, p, label=exp)
plt.ylabel("CDF", fontsize=25)
plt.xlabel("Latency (ms)", fontsize=25)
# plt.legend(bbox_to_anchor=(1.02, 1))
# plt.legend()
# plt.xlim([0, 101])
plt.ylim([0, 1])
plt.legend(edgecolor='black', loc='upper center', ncol=3, fancybox=True, framealpha=.2)
plt.savefig("fb-lat.png",bbox_inches='tight')