hillclimbing_result = {}

# from unittest import result
import pickle

for f in os.listdir("/Users/janechen/Downloads/darwin-data/results-hillclimbing"):
    dict = pickle.load(open("/Users/janechen/Downloads/darwin-data/results-hillclimbing/"+f, "rb"))
    # print(dict)
    # break
    for trace in dict.keys():
        if trace in hillclimbing_result.keys():
        #     print(trace)
            hillclimbing_result[trace].update(dict[trace])
        else:
            hillclimbing_result[trace] = dict[trace]
    hillclimbing_result.update(dict)
# print(result)