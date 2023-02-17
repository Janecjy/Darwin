import os
import pickle

input_dir = "/mydata/output-offline-3d/"

ohr = {}

def addToMap(dst, src_f):
    sub_result = pickle.load(open(src_f, "rb"))
    dst.update(sub_result)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.startswith("ohr"):
            addToMap(ohr, root+"/"+file)
        # print(ohr)
        # exit(0)

pickle.dump(ohr, open(input_dir+"ohr.pkl", "wb"))

# ohr = pickle.load(open(input_dir+"ohr.pkl", "rb"))

best_count = {} # expert: best count

for trace, results in ohr.items():
    if len(results) == 0:
        continue
    best_expert = max(results, key=results.get)
    if best_expert not in best_count.keys():
        best_count[best_expert] = 0
    best_count[best_expert] += 1

print(best_count)

for thres in [1, 2, 3, 4, 5]:
    coarse_best_result = {}
    for trace, ohr_results in ohr.items():
        if len(ohr_results) == 0:
            continue
        # assert ohr_results, f"({trace}, {ohr_results})"
        coarse_best_result[trace] = []
        results = {}
        for exp in ohr_results.keys():
            assert exp in ohr_results
            results[exp] = ohr_results[exp]
        min_result = min(results.values())
        for exp in results.keys():
            if results[exp] - min_result < thres:
                coarse_best_result[trace].append(exp)
    pickle.dump(coarse_best_result, open("/mydata/results/3d_coarse_best_result_"+str(thres)+".pkl", "wb"))
