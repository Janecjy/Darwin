import os
import pickle

input_dir = "/mydata/results-offline/"

# bmr = {}
# dw = {}
# ohr = {}

# def addToMap(dst, src_f):
#     sub_result = pickle.load(open(src_f, "rb"))
#     dst.update(sub_result)

# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         if file.startswith("bmr"):
#             addToMap(bmr, root+"/"+file)
#         if file.startswith("dw"):
#             addToMap(dw, root+"/"+file)
#         if file.startswith("ohr"):
#             addToMap(ohr, root+"/"+file)
#         # print(bmr)
#         # print(dw)
#         # print(ohr)
#         # exit(0)

# # assert(len(bmr.keys()) == 700), len(bmr.keys())
# # assert(len(dw.keys()) == 700), len(dw.keys())
# # assert(len(ohr.keys()) == 700), len(ohr.keys())
# pickle.dump(bmr, open(input_dir+"bmr.pkl", "wb"))
# pickle.dump(dw, open(input_dir+"dw.pkl", "wb"))
# pickle.dump(ohr, open(input_dir+"ohr.pkl", "wb"))


# bmr = pickle.load(open(input_dir+"bmr.pkl", "rb"))
# # del bmr['script']
# # pickle.dump(bmr, open(input_dir+"bmr.pkl", "wb"))

# best_count = {} # expert: best count

# for trace, results in bmr.items():
#     assert results, f"({trace}, {results})"
#     best_expert = min(results, key=results.get)
#     if best_expert not in best_count.keys():
#         best_count[best_expert] = 0
#     best_count[best_expert] += 1

# print(best_count)


# ohr = pickle.load(open(input_dir+"ohr.pkl", "rb"))
# # del bmr['script']
# # pickle.dump(bmr, open(input_dir+"bmr.pkl", "wb"))

# best_count = {} # expert: best count

# for trace, results in ohr.items():
#     assert results, f"({trace}, {results})"
#     best_expert = min(results, key=results.get)
#     if best_expert not in best_count.keys():
#         best_count[best_expert] = 0
#     best_count[best_expert] += 1

# print(best_count)

dw = pickle.load(open(input_dir+"dw.pkl", "rb"))
ohr = pickle.load(open(input_dir+"ohr.pkl", "rb"))
# del dw['script']
# del ohr['script']
# pickle.dump(dw, open(input_dir+"dw.pkl", "wb"))
# pickle.dump(ohr, open(input_dir+"ohr.pkl", "wb"))

best_count = {} # expert: best count

for trace, dw_results in dw.items():
    assert dw_results, f"({trace}, {dw_results})"
    ohr_results = ohr[trace]
    results = {}
    for exp in dw_results.keys():
        assert exp in ohr_results
        results[exp] = ohr_results[exp] - dw_results[exp]/(9*10e6)
        # print(ohr_results[exp], dw_results[exp]/(9*10e6), results[exp])
    best_expert = max(results, key=results.get)
    if best_expert not in best_count.keys():
        best_count[best_expert] = 0
    best_count[best_expert] += 1

print(best_count)

for thres in [1, 2, 3, 4, 5]:
    coarse_best_result = {}
    for trace, dw_results in dw.items():
        assert dw_results, f"({trace}, {dw_results})"
        coarse_best_result[trace] = []
        ohr_results = ohr[trace]
        results = {}
        for exp in dw_results.keys():
            assert exp in ohr_results
            results[exp] = ohr_results[exp] - dw_results[exp]/(9*10e6)
        max_result = max(results.values())
        for exp in results.keys():
            if max_result - results[exp] < thres:
                coarse_best_result[trace].append(exp)
    pickle.dump(coarse_best_result, open("/mydata/results/oh_dw_coarse_best_result_"+str(thres)+".pkl", "wb"))
