'''
print the correct rate in test traces with the given accuracy threshold
'''
def printcorrect(expert0, expert1, hidden, accuracy_thres):
    e0 = []
    e1_r = []
    e1_p = []
    correct = 0
    start = False
    for line in open("/mydata/models/"+expert0+'-'+expert1+'/'+str(hidden)+".result"):
        if "=====Epoch 99" in line:
            start = True
        if start:
            if "e0 hit rate" in line:
                e0.append(float(line.split()[3]))
            if "e1 real hit rate" in line:
                e1_r.append(float(line.split()[4]))
            if "e1 predicted hit rate" in line:
                e1_p.append(float(line.split()[4]))
    assert len(e0) == len(e1_r) == len(e1_p)
    for i, j, k in zip(e0, e1_r, e1_p):
        if abs(i-j) < accuracy_thres or (i < j and i < k) or (i >= j and i >= k):
            correct += 1
    return correct/len(e0)

# get correct rate with accuracy threshold 1% for all the models
import os
results = []
for f0 in [2, 3, 4, 5, 6, 7]:
# for f0 in [2]:
    for s0 in [10, 20, 50, 100, 500, 1000]:
        for f1 in [2, 3, 4, 5, 6, 7]:
            for s1 in [10, 20, 50, 100, 500, 1000]:
                if (f0 != f1 or s0 != s1) and os.path.exists("/mydata/models/"+'f'+str(f0)+'s'+str(s0)+'-f'+str(f1)+'s'+str(s1)+'/model-h2.ckpt'):
                    try:
                        c = printcorrect('f'+str(f0)+'s'+str(s0), 'f'+str(f1)+'s'+str(s1), 2, 1)
                        if c < 0.8:
                            print('f'+str(f0)+'s'+str(s0)+'-f'+str(f1)+'s'+str(s1))
                        results.append(c)
                    except:
                        continue

'''
print percent of test data points are correct because e0 and e1 are closer than accuracy thresholds
'''        
def printpercent(expert0, expert1, hidden, accuracy_thres):
    e0 = []
    e1_r = []
    e1_p = []
    strict_correct = 0
    correct = 0
    start = False
    for line in open("/mydata/models/"+expert0+'-'+expert1+'/'+str(hidden)+".result"):
        if "=====Epoch 99" in line:
            start = True
        if start:
            if "e0 hit rate" in line:
                e0.append(float(line.split()[3]))
            if "e1 real hit rate" in line:
                e1_r.append(float(line.split()[4]))
            if "e1 predicted hit rate" in line:
                e1_p.append(float(line.split()[4]))
    assert len(e0) == len(e1_r) == len(e1_p)
    for i, j, k in zip(e0, e1_r, e1_p):
        if abs(i-j) < accuracy_thres:
            correct += 1
    return correct/len(e0)


# plot accuracy for all models
import matplotlib.pyplot as plt
# for i in range(len(results)):
plt.figure(figsize=(12, 5))
plt.plot(range(len(results)), strict_results, label = "Strict Accuracy")
plt.plot(range(len(results)), results, label = "1% Accuracy")
plt.plot(range(len(results)), results_2, label = "2% Accuracy")
plt.plot(range(len(results)), results_3, label = "3% Accuracy")
# plt.plot(range(len(strict_ood)), strict_ood, label = "OOD Strict Accuracy")
# plt.plot(range(len(results_ood_1)), results_ood_1, label = "OOD 1% Accuracy")
# plt.plot(range(len(results_ood_2)), results_ood_2, label = "OOD 2% Accuracy")
# plt.plot(range(len(results_ood_3)), results_ood_3, label = "OOD 3% Accuracy")
# plt.xlim([0, 50])
plt.legend()
plt.show()

# plot accuracy compared with ood
import matplotlib.pyplot as plt
# for i in range(len(results)):
plt.figure(figsize=(12, 5))
plt.plot(range(len(results)), strict_results, label = "Strict Accuracy")
plt.plot(range(len(strict_ood)), strict_ood, label = "OOD Strict Accuracy")
plt.xlim([0, len(strict_ood)-1])
plt.xlabel("Model Num")
plt.ylabel("Prediction Accuracy")
plt.legend()
plt.show()

# plot accuracy cdf
import matplotlib.pyplot as plt
import numpy as np


# sort the data:
strict_sorted = np.sort(strict_results)
sorted = np.sort(results)

# calculate the proportional values of samples
p = 1. * np.arange(len(strict_sorted)) / (len(strict_sorted) - 1)

# plot the sorted data:
fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax1.plot(p, strict_sorted)
# ax1.set_xlabel('$p$')
# ax1.set_ylabel('$x$')

# ax2 = fig.add_subplot(122)
plt.plot(strict_sorted, p, label = "Strict Accuracy")
plt.plot(sorted, p, label = "1% Accuracy")
plt.legend()
plt.show()
# plt.set_xlabel('$x$')
# plt.set_ylabel('$p$')

def printcorrect_debug(expert0, expert1, hidden, accuracy_thres):
    e0 = []
    e1_r = []
    e1_p = []
    strict_correct = 0
    correct = 0
    start = False
    for line in open("/mydata/models/"+expert0+'-'+expert1+'/'+str(hidden)+"-result.out"):
        if "=====Epoch 99" in line:
            start = True
        if start:
            if "e0 hit rate" in line:
                e0.append(float(line.split()[3]))
            if "e1 real hit rate" in line:
                e1_r.append(float(line.split()[4]))
            if "e1 predicted hit rate" in line:
                e1_p.append(float(line.split()[4]))
    assert len(e0) == len(e1_r) == len(e1_p)
    count = 0
    x_correct = []
    e0_correct = []
    e1_correct = []
    e1_p_correct = []
    x_wrong = []
    e0_wrong = []
    e1_wrong = []
    e1_p_wrong = []
    for i, j, k in zip(e0, e1_r, e1_p):
        if (i < j and i < k) or (i >= j and i >= k):
            strict_correct += 1
        if abs(i-j) < accuracy_thres or (i < j and i < k) or (i >= j and i >= k):
            correct += 1
            x_correct.append(count)
            e0_correct.append(i)
            e1_correct.append(j)
            e1_p_correct.append(k)
        else:
            x_wrong.append(count)
            e0_wrong.append(i)
            e1_wrong.append(j)
            e1_p_wrong.append(k)
        #     print(count, i, j, k)
        count += 1
    return strict_correct/len(e0), correct/len(e0), x_correct, e0_correct, e1_correct, x_wrong, e0_wrong, e1_wrong, e1_p_correct, e1_p_wrong
    # print(strict_correct/len(e0), correct/len(e0))

s, c, x_correct, e0_correct, e1_correct, x_wrong, e0_wrong, e1_wrong, e1_p_correct, e1_p_wrong = printcorrect_debug("f2s20", "f3s50", 2, 2)

# plot the e0, e1 correct and wrong hit rates
import matplotlib.pyplot as plt
# for i in range(len(results)):
plt.figure(figsize=(20, 10))
plt.scatter(x_correct, e0_correct, s=2, label = "Correct Prediction E0 Hit Rate")
plt.scatter(x_correct, e1_correct, s=2, label = "Correct Prediction E1 Hit Rate")
# plt.scatter(x_correct, e1_p_correct, s=1.5, label = "Correct Prediction E1 Predicted Hit Rate")
plt.scatter(x_wrong, e0_wrong, s=2, label = "Wrong Prediction E0 Hit Rate")
plt.scatter(x_wrong, e1_wrong, s=2, label = "Wrong Prediction E1 Hit Rate")
# plt.scatter(x_wrong, e1_p_wrong, s=1.5, label = "Wrong Prediction E1 Predicted Hit Rate")
plt.legend()
plt.xlim([0, 100])
plt.show()


# original method to get the train data according to offline results

import pickle
expert_0 = "f2s10"
expert_1 = "f2s20"
name = "tc-0-1-0:265-0"
bucket_list = [10, 20, 50, 100, 500, 1000, 5000]
ROUND_LEN = 500000

e0_hits = pickle.load(open(os.path.join("/mydata/experts/", expert_0, name+'.pkl'), "rb"))
e1_hits = pickle.load(open(os.path.join("/mydata/experts/", expert_1, name+'.pkl'), "rb"))

hit_hit_prob = 0 # pi(e1_hit | e0_hit)
e0_hit_count = 0
e0_miss_count = 0
hit_miss_prob = 0 # pi(e1_hit | e0_miss)
bucket_count = [0]*(len(bucket_list)+1)

assert (len(e0_hits) == len(e1_hits))

count = 0
# import pdb; pdb.set_trace()
for e0, e1 in zip(e0_hits, e1_hits):
    if count > 0 and count % ROUND_LEN == 0:
        hit_hit_prob = hit_hit_prob/e0_hit_count
        hit_miss_prob = hit_miss_prob/e0_miss_count
        print(hit_hit_prob, hit_miss_prob)
        print(bucket_count)
        e0_hit_count = e0_miss_count = hit_hit_prob = hit_miss_prob = 0
        bucket_count = [0]*(len(bucket_list)+1)
        # break
    # else:
        # if count == TRAINING_LEN:
    if e0[1] == 1:
        e0_hit_count += 1
        if e1[1] == 1:
            hit_hit_prob += 1
    else:
        e0_miss_count += 1
        if e1[1] == 1:
            hit_miss_prob += 1
            
    for j in range(len(bucket_list)):
        if e1[0] < bucket_list[j]:
            bucket_count[j] += 1
            # print(count, j)
            break
    if e1[0] >= bucket_list[-1]:
        bucket_count[-1] += 1
        # print(count, len(bucket_list))
    
        # break
            # hit_hit_prob = e0_hit_count = e0_miss_count = hit_miss_prob = 0
        # if name not in test_files:
        #     break
        # if count >= TRAINING_LEN + PREDICTION_LEN:
        #     prediction_input.append([feature, e0_hit_count, e0_miss_count])
        #     hit_hit_prob = hit_hit_prob/e0_hit_count
        #     hit_miss_prob = hit_miss_prob/e0_miss_count
        #     prediction_labels.append([hit_hit_prob, hit_miss_prob])
        #     break
        # if e0 == 1:
        #     e0_hit_count += 1
        #     if e1 == 1:
        #         hit_hit_prob += 1
        # else:
        #     e0_miss_count += 1
        #     if e1 == 1:
        #         hit_miss_prob += 1
    count += 1
if count > 0 and count % ROUND_LEN == 0:
    hit_hit_prob = hit_hit_prob/e0_hit_count
    hit_miss_prob = hit_miss_prob/e0_miss_count
    print(hit_hit_prob, hit_miss_prob)
    print(bucket_count)
    

# alternative method with logical operations

import pickle
import numpy as np
expert_0 = "f2s10"
expert_1 = "f2s20"
name = "tc-0-1-0:265-0"
bucket_list = [0, 10, 20, 50, 100, 500, 1000, 5000, 100000000]
ROUND_LEN = 500000

e0_hits = np.array([*pickle.load(open(os.path.join("/mydata/experts/", expert_0, name+'.pkl'), "rb"))])
e1_hits = np.array([*pickle.load(open(os.path.join("/mydata/experts/", expert_1, name+'.pkl'), "rb"))])

hit_hit_prob = 0 # pi(e1_hit | e0_hit)
e0_hit_count = 0
e0_miss_count = 0
hit_miss_prob = 0 # pi(e1_hit | e0_miss)
bucket_count = [0]*(len(bucket_list)+1)

assert (len(e0_hits) == len(e1_hits))

# for i in range(len(e0_hits))
r, c = e0_hits.shape
for i in range(r//ROUND_LEN):
    e0 = e0_hits[ROUND_LEN*i:ROUND_LEN*(i+1), 1]
    e1 = e1_hits[ROUND_LEN*i:ROUND_LEN*(i+1), 1]
    s0 = e0_hits[ROUND_LEN*i:ROUND_LEN*(i+1), 0]
    hit_hit_prob = np.sum(np.logical_and(e0, e1))/np.sum(e0)
    hit_miss_prob = 1-np.sum(np.logical_and(np.logical_not(e0), np.logical_not(e1)))/(ROUND_LEN-np.sum(e0))
    print(hit_hit_prob, hit_miss_prob)
    digitized = np.digitize(s0, bucket_list)
    bucket_count = [len(s0[digitized == i]) for i in range(1, len(bucket_list))]
    print(bucket_count)
