import re
import os
import pickle

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def countStat(dirPath):
    trace_stats = {}
    current_root = None
    
    for root, dirs, files in os.walk(dirPath):
        
        for file in files:
            if not file.endswith(".out"):
                continue
            trace = root.split('/')[-1]

            if trace not in trace_stats.keys():
                trace_stats[trace] = {}
                
            file_res = []

            for line in open(os.path.join(root, file), "r"):
                
                val2 = re.findall('hoc hit: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                
                for sentence in val2:
                    exp = sentence.split(',')
                    exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                    file_res.append(exp)

            if file_res:
                
                if file.endswith(".out"):
                    trace_stats[trace]["online"] = [x[0] for x in file_res][-1]
    print(trace_stats)  
    pickle.dump(trace_stats, open("/mydata/output-online/results.pkl", "wb"))
    
if __name__ == "__main__":
    countStat("/mydata/output-online/")