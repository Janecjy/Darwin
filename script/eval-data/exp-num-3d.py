import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

exp_num = pickle.load(open("/mydata/results/exp-num-3d.pkl", "rb"))
print(exp_num)

sns.set(font_scale=1.5, style='white')
plt.figure(figsize=(10, 10))
thres = [1, 2, 3, 4, 5]
for t in thres:
    exp_num = pickle.load(open("/mydata/results/exp-num-3d-"+str(t)+".pkl", "rb"))
    sorted_len = np.sort(exp_num)
    p = 1. * np.arange(len(sorted_len)) / (len(sorted_len) - 1)
    plt.plot(sorted_len, p, label=str(t)+'%', linewidth=4)
    plt.ylabel("CDF", fontsize=25)
    plt.xlabel("Remaining Expert Number", fontsize=25)
    # plt.legend(bbox_to_anchor=(1.02, 1))
    plt.legend()
    # plt.xlim([0, 37])
    plt.ylim([0, 1])
    plt.savefig("exp-num.png",bbox_inches='tight')
    if t == 2:
        for i, cdf in enumerate(p):
            if cdf > 0.9:
                print(sorted_len[i])