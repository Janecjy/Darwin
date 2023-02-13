import pickle
import os

input_dir = "/mydata/features-online/"
features_list = ['iat_avg', 'sd_avg', 'size_avg']
comparison_length = 50

output = {}
for trace in os.listdir(input_dir):
    # print(trace)
    if not trace.endswith("pkl"):
        trace_features = {}
        for file in os.listdir(input_dir+trace):
            collection_length = int(file.split('M')[0])
            input = pickle.load(open(input_dir+trace+"/"+file, "rb"))
            features = []
            for feat in features_list:
                if type(input[feat]) is dict:
                    for k, v in sorted(input[feat].items()):
                        features.append(v)
                else:
                    features.append(input[feat])
            # print(features)
            trace_features[collection_length] = features
        output[trace] = trace_features

pickle.dump(output, open("/mydata/features-online/features.pkl", "wb"))


output = pickle.load(open("/mydata/features-online/features.pkl", "rb"))

features_list = []
for i in range(7):
    features_list.append('iat_avg'+str(i))
for i in range(7):
    features_list.append('sd_avg'+str(i))
features_list.append('size_avg')

diff_output = {}
for feature in features_list:
    diff_output[feature] = {}
    for length in range(comparison_length):
        if length > 0:
            diff_output[feature][length] = []
for trace in output.keys():
    if output[trace]:
        for i, feature in enumerate(features_list):
            # print(trace, output[trace])
            goal = output[trace][comparison_length][i]
            for length in range(comparison_length):
                if length > 0:
                    diff_output[feature][length].append((output[trace][length][i] - goal) /goal*100)

for feature in features_list:
    # diff_output[feature] = {}
    for length in range(comparison_length):
        if length > 0:
            # print(feature, length, diff_output[feature])
            avg = sum(diff_output[feature][length])/len(diff_output[feature][length])
            diff_output[feature][length] = abs(avg)

pickle.dump(diff_output, open("/mydata/features-online/feature_diff.pkl", "wb"))


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.5, style='white')

fig = plt.figure()
markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
diff_output = pickle.load(open("/mydata/features-online/feature_diff.pkl", "rb"))
# for i, length in enumerate([1, 2, 3, 5, 10, 20, 30, 40]):
if comparison_length == 10:
    length_list = range(10)
else: 
    length_list = [1, 2, 3, 5, 10, 20, 30, 40]
for i, length in enumerate(length_list):
    if length > 0:
        data = []
        for feature in features_list:
            # print(diff_output[feature][length])
            data.append(diff_output[feature][length])
        if length == 3:
            size = 50
        else:
            size = 5
        if comparison_length == 10:
            plt.scatter(x=range(len((data))), y=data, s=size, label=str(length)+'M', marker=markers[length])
        else:
            plt.scatter(x=range(len((data))), y=data, s=size, label=str(length)+'M', marker=markers[i+1])
        if length == 3:
            print(data)
plt.xticks(range(len((data))), features_list, rotation=70, fontsize=15)
plt.xlabel("Feature", fontsize=15)
plt.ylabel("Difference (%)", fontsize=15)
plt.legend(edgecolor='black',fontsize=15,bbox_to_anchor=(1.02, 1))
plt.savefig("feature-convergence.png",bbox_inches='tight')