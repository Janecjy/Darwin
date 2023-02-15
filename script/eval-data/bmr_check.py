import os
import pickle

input_dir = "/mydata/results-offline/"

bmr = pickle.load(open(input_dir+"bmr.pkl", "rb"))

best_count = {} # expert: best count

for trace, results in bmr.items():
    assert results, f"({trace}, {results})"
    best_expert = min(results, key=results.get)
    if best_expert not in best_count.keys():
        best_count[best_expert] = 0
    best_count[best_expert] += 1

print(best_count)

for thres in [1, 2, 3, 4, 5]:
    coarse_best_result = {}
    for trace, bmr_results in bmr.items():
        assert bmr_results, f"({trace}, {bmr_results})"
        coarse_best_result[trace] = []
        results = {}
        for exp in bmr_results.keys():
            assert exp in bmr_results
            results[exp] = bmr_results[exp]
        min_result = min(results.values())
        for exp in results.keys():
            if results[exp] - min_result < thres:
                coarse_best_result[trace].append(exp)
    pickle.dump(coarse_best_result, open("/mydata/results/oh_dw_coarse_best_result_"+str(thres)+".pkl", "wb"))
