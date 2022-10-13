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

            if not file.endswith("-6exp.out") and not file.endswith("-6exp-5k.out") and not file.endswith(".txt"):
                continue
            
            if file.endswith("-6exp-5k.out"):
                # online 6 expert output files
                trace = '-'.join(file.split('-')[:-2])
            if file.endswith("-6exp.out"):
                trace = '-'.join(file.split('-')[:-1])
            if file.endswith(".txt"):
                trace = root.split('/')[-1]

            if trace not in trace_stats.keys():
                trace_stats[trace] = {}
                
            file_res = []

            for line in open(os.path.join(root, file), "r"):
                val1 = re.findall('freq: [\d]*, size: [\d]*',line)

                for sentence in val1:
                    sentence = sentence.split(',')
                    f = (sentence[0].split(':')[1].replace(" ", ""))
                    s = (sentence[1].split(':')[1].replace(" ", ""))
                
                val2 = re.findall('hoc hit: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                
                for sentence in val2:
                    exp = sentence.split(',')
                    exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                    file_res.append(exp)

            if file_res:
                if file.endswith("-6exp.out"):
                    trace_stats[trace]['6exp-online'] = [x[0] for x in file_res][-1]
                if file.endswith("-6exp-5k.out"):
                    trace_stats[trace]['6exp-online-5k'] = [x[0] for x in file_res][-1]
                if file.endswith(".txt"):
                    trace_stats[trace]['f'+f+'s'+s] = [x[0] for x in file_res][-1]
    print(trace_stats)  
    pickle.dump(trace_stats, open("/mydata/output/results.pkl", "wb"))
    
if __name__ == "__main__":
    countStat("/mydata/output/")