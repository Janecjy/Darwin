import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5, style='white')

dir_path = sys.argv[1]
converge_times = 5
round_length = []
for f in os.listdir(dir_path):
    if not f.endswith(".txt"):
        continue
    start = False
    finish = False
    potential_length = 0
    history = []
    converge = False
    for line in open(dir_path+f):
        if line.startswith("cluster predicted best expert:"):
            start = True
        if start and line.startswith("['f"):
            potential_length = len(line.split(','))
        if start and not finish and line.startswith("{'f"):
            values = []
            for l in line.split(','):
                values.append(float(l.split(':')[1].replace("}\n", "")))
            # print(values)
            history.append(np.argmax(values))
        if len(history) > converge_times:
            converge = True
            for j in range(converge_times):
                if history[-j-1] != history[-j-2]:
                    converge = False
        if converge:
            print(len(history))
            round_length.append(len(history))
            break
        if line.startswith("selected times:"):
            finish = True

plt.figure(figsize=(10, 10))
sorted_len = np.sort(round_length)
p = 1. * np.arange(len(sorted_len)) / (len(sorted_len) - 1)
plt.plot(sorted_len, p, linewidth=4)
plt.ylabel("CDF", fontsize=25)
plt.xlabel("Number of Rounds", fontsize=25)
# plt.legend(bbox_to_anchor=(1.02, 1))
# plt.legend()
# plt.xlim([0, 101])
plt.ylim([0, 1])
plt.savefig("bandit-round-num.png",bbox_inches='tight')