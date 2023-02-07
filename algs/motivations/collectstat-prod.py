import os
import re
import pickle
# parse output data
input_dir = "/mydata/output-production/"

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def countStat(dirPath):
    trace_ohr = {}
    trace_dw = {}
    
    for dir in os.listdir(input_dir):
        if dir != "eu-subtraces":
        
            for file in os.listdir(input_dir+dir):
                if file.endswith(".txt"):
                    trace = dir
                        
                    ohr_res = []
                    dw_res = []
                    
                    if trace not in trace_ohr.keys():
                        trace_ohr[trace] = {}
                        trace_dw[trace] = {}

                    for line in open(os.path.join(input_dir+dir+"/"+file), "r"):
                        val1 = re.findall('freq: [\d]*, size: [\d]*',line)

                        for sentence in val1:
                            sentence = sentence.split(',')
                            f = (sentence[0].split(':')[1].replace(" ", ""))
                            s = (sentence[1].split(':')[1].replace(" ", ""))
                        
                        val2 = re.findall('hoc hit: [\d]+[.]?[\d]*%',line)
                        
                        for sentence in val2:
                            exp = float(sentence.replace(" ", "").replace("%", "").split(':')[1])
                            ohr_res.append(exp)
                        
                        val3 = re.findall('disk write: [\d]+[.]?[\d]*', line)
                        for sentence in val3:
                            exp = float(sentence.replace(" ", "").split(':')[1])
                            dw_res.append(exp)

                    if ohr_res:
                        # print(ohr_res)
                        if file.endswith(".txt"):
                            trace_ohr[trace]['f'+str(f)+'s'+str(s)] = ohr_res[-1]
                            trace_dw[trace]['f'+str(f)+'s'+str(s)] = dw_res[-1]
    print(trace_ohr) 
    print(trace_dw)  
    pickle.dump(trace_ohr, open("/mydata/output-production/"+trace+"/ohr-result.pkl", "wb"))
    pickle.dump(trace_dw, open("/mydata/output-production/"+trace+"/dw-result.pkl", "wb"))


for dir in os.listdir(input_dir):
    print(dir)
    if dir != "figs":
        countStat(input_dir+dir)
# draw ohr

# draw disk write