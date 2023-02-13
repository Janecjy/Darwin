# '''
# print the correct rate in test traces with the given accuracy threshold
# '''
# def printcorrect(expert0, expert1, hidden, accuracy_thres):
#     e0 = []
#     e1_r = []
#     e1_p = []
#     correct = 0
#     start = False
#     for line in open("/mydata/nn-models/"+expert0+'-'+expert1+'/'+str(hidden)+".result"):
#         if "=====Epoch 99" in line:
#             start = True
#         if start:
#             if "e0 hit rate" in line:
#                 e0.append(float(line.split()[3]))
#             if "e1 real hit rate" in line:
#                 e1_r.append(float(line.split()[4]))
#             if "e1 predicted hit rate" in line:
#                 e1_p.append(float(line.split()[4]))
#     assert len(e0) == len(e1_r) == len(e1_p)
#     for i, j, k in zip(e0, e1_r, e1_p):
#         if abs(i-j) < accuracy_thres or (i < j and i < k) or (i >= j and i >= k):
#             correct += 1
#     return correct/len(e0)

# def printoodcorrect(expert0, expert1, hidden, accuracy_thres):
#     e0 = []
#     e1_r = []
#     e1_p = []
#     correct = 0
#     start = False
#     for line in open("/mydata/nn-models/"+expert0+'-'+expert1+'/'+str(hidden)+"-ood-result.out"):
#         if "=====Epoch 99" in line:
#             start = True
#         if start:
#             if "e0 hit rate" in line:
#                 e0.append(float(line.split()[3]))
#             if "e1 real hit rate" in line:
#                 e1_r.append(float(line.split()[4]))
#             if "e1 predicted hit rate" in line:
#                 e1_p.append(float(line.split()[4]))
#     assert len(e0) == len(e1_r) == len(e1_p)
#     for i, j, k in zip(e0, e1_r, e1_p):
#         if abs(i-j) < accuracy_thres or (i < j and i < k) or (i >= j and i >= k):
#             correct += 1
#     return correct/len(e0)

# # get correct rate with accuracy threshold 1% for all the models
# import os
# import pickle
# results_1 = []
# results_1_ood = []
# results_2 = []
# results_2_ood = []
# results_3 = []
# results_3_ood = []

# for f0 in [2, 3, 4, 5, 6, 7]:
# # for f0 in [2]:
#     for s0 in [10, 20, 50, 100, 500, 1000]:
#         for f1 in [2, 3, 4, 5, 6, 7]:
#             for s1 in [10, 20, 50, 100, 500, 1000]:
#                 if (f0 != f1 or s0 != s1) and os.path.exists("/mydata/nn-models/"+'f'+str(f0)+'s'+str(s0)+'-f'+str(f1)+'s'+str(s1)+'/model-h2.ckpt'):
#                     try:
#                         c = printcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 1)
#                         # if c < 0.8:
#                         #     print('f'+str(f0)+'s'+str(s0)+'-f'+str(f1)+'s'+str(s1))
#                         results_1.append(c)
#                         c = printcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 2)
#                         results_2.append(c)
#                         c = printcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 3)
#                         results_3.append(c)
#                         c = printoodcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 1)
#                         results_1_ood.append(c)
#                         c = printoodcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 2)
#                         results_2_ood.append(c)
#                         c = printoodcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 3)
#                         results_3_ood.append(c)
#                     except:
#                         continue

# pickle.dump(results_1, open("/mydata/results/nn_results_1.pkl", "wb"))
# pickle.dump(results_2, open("/mydata/results/nn_results_2.pkl", "wb"))
# pickle.dump(results_3, open("/mydata/results/nn_results_3.pkl", "wb"))
# pickle.dump(results_1_ood, open("/mydata/results/nn_results_1_ood.pkl", "wb"))
# pickle.dump(results_2_ood, open("/mydata/results/nn_results_2_ood.pkl", "wb"))
# pickle.dump(results_3_ood, open("/mydata/results/nn_results_3_ood.pkl", "wb"))


import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(font_scale=1.5, style='white')

results_1 = pickle.load(open("/mydata/results/nn_results_1.pkl", "rb"))
results_2 = pickle.load(open("/mydata/results/nn_results_2.pkl", "rb"))
results_3 = pickle.load(open("/mydata/results/nn_results_3.pkl", "rb"))
results_1_ood = pickle.load(open("/mydata/results/nn_results_1_ood.pkl", "rb"))
results_2_ood = pickle.load(open("/mydata/results/nn_results_2_ood.pkl", "rb"))
results_3_ood = pickle.load(open("/mydata/results/nn_results_3_ood.pkl", "rb"))

plt.figure()
labels = ["1% Accuracy", "1% OOD Accuracy", "2% Accuracy", "2% OOD Accuracy", "3% Accuracy", "3% OOD Accuracy"]
for i, result in enumerate([results_1, results_1_ood, results_2, results_2_ood, results_3, results_3_ood]):
    sorted_len = np.sort(result)
    p = 1. * np.arange(len(sorted_len)) / (len(sorted_len) - 1)
    plt.plot([x*100 for x in sorted_len], p, label=labels[i])
plt.ylabel("CDF")
plt.xlabel("Accuracy (%)")
# plt.legend(bbox_to_anchor=(1.02, 1))
plt.legend()
plt.xlim([0, 101])
plt.ylim([0, 1])
plt.savefig("nnmodel-accuracy.png",bbox_inches='tight')