import pickle
import sys

def loadDataIter(dirPath, iter):
    data = pickle.load(open(dirPath+str(iter)+"M.pkl", "rb"))
    res = {}
    for key in data.keys() & {'sd_avg', 'iat_avg', 'size_avg', 'edc_avg'}:
        # print(data[key])
        if key == 'size_avg':
            res[key] = data[key]
        else:
            for num_key in data[key].keys():
                res[key+'_'+str(num_key)] = data[key][num_key]
    return res

def loadData(dirPath):
    res_list = []
    for i in range(9):
        res_list.append(loadDataIter(dirPath, i+1))
    return res_list

# result format: {feature: [[1M diff], [2M diff], ...]}
def computeDiff(res_list):
    result = {}
    for key in res_list[0].keys():
        result[key] = []
        for i in range(len(res_list)-1):
            result[key].append((res_list[i][key]-res_list[len(res_list)-1][key])/res_list[len(res_list)-1][key]*100)
    return result

def main():
    dirPath = sys.argv[1]
    res_list = loadData(dirPath)
    result = computeDiff(res_list)
    pickle.dump(result, open("/mydata/featurediff/"+dirPath.split('/')[3]+".pkl", "wb"))

if __name__ == '__main__':
    main()
    