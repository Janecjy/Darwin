import os
import sys
import pickle

input_dir = sys.argv[1]
output_dir = sys.argv[2]

ohr = {}

# def addToMap(dst, src_f):
#     sub_result = {}
#     sub_result = pickle.load(open(src_f, "rb"))
#     dst.update(sub_result)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith("txt") and not file.endswith("-hits.txt"):
            # addToMap(ohr, root+"/"+file)
            trace_name = root.split("/")[-1]
            expert_name = ''.join(file.split(".")[0].split("-"))
            if trace_name not in ohr.keys():
                ohr[trace_name] = {}
            with open(root+"/"+file, 'r') as f:
                for line in f:
                    if (line.startswith("final hoc hit")):
                        v = float(line.split("%")[0].split(":")[1].strip())
            ohr[trace_name][expert_name] = v
                        # print(trace_name, expert_name, ohr)
                        # exit(0)
            # print(trace_name, expert_name)
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
        max_result = max(results.values())
        for exp in results.keys():
            if max_result - results[exp] < thres:
                coarse_best_result[trace].append(exp)
    pickle.dump(coarse_best_result, open(os.path.join(output_dir, "coarse_best_result_"+str(thres)+".pkl"), "wb"))
