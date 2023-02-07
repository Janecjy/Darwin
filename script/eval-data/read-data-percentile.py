import os

percentile_result = {}

# from unittest import result
import pickle

for f in os.listdir("/Users/janechen/Downloads/darwin-data/results-percentile-new"):
    dict = pickle.load(open("/Users/janechen/Downloads/darwin-data/results-percentile-new/"+f, "rb"))
    # if "tc-0-1-228:37-0" in dict.keys():
    #     print(f)
        # print(dict)
    for trace in dict.keys():
        if trace in percentile_result.keys():
        #     print(trace)
            percentile_result[trace].update(dict[trace])
        else:
            percentile_result[trace] = dict[trace]
    percentile_result.update(dict)
# print(result)

# check if percentile results are complete

# trace = percentile_result.keys()[0]
percentile_length = len(list(percentile_result.values())[1])
print(percentile_length)
for k, v in percentile_result.items():
    if len(v) != percentile_length:
        print(k, len(v))


# find best configuration for percentile

best_counts = {} # conf: count
for trace in percentile_result.keys():
    result = percentile_result[trace]
    if result:
        best_conf = max(result, key=result.get)
        if best_conf not in best_counts.keys():
            best_counts[best_conf] = 0
        best_counts[best_conf] += 1
print(best_counts)

# f6s9l100000 is the best