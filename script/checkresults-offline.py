import re
import os
import pickle

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def countStat(dirPath):
    ohr = {}
    # bmr = {}
    # dw = {}
    current_root = None
    
    for root, dirs, files in os.walk(dirPath):
        
        for file in files:
            if file.endswith("hits.txt") or not file.endswith("txt"):
                continue
            trace = root.split('/')[-1]

            if trace not in ohr.keys():
                ohr[trace] = {}
                # bmr[trace] = {}
                # dw[trace] = {}
            
            f = file.split('-')[0].replace("f", "")
            s = file.split('-')[1].split('.')[0].replace("s", "")
                
            file_res = []

            for line in open(os.path.join(root, file), "r"):
                
                val2 = re.findall('final hoc hit: [\d]+[.]?[\d]*%, hoc byte miss: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                
                for sentence in val2:
                    exp = sentence.split(',')
                    exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                    file_res.append(exp)

            if file_res:
                
                # if file.endswith(".out"):
                ohr[trace]["f"+f+"s"+s] = [x[0] for x in file_res][-1]
                # bmr[trace]["f"+f+"s"+s] = [x[1] for x in file_res][-1]
                # dw[trace]["f"+f+"s"+s] = [x[5] for x in file_res][-1]

    pickle.dump(ohr, open(dirPath+"ohr.pkl", "wb"))
    # pickle.dump(bmr, open("/mydata/results-offline/bmr.pkl", "wb"))
    # pickle.dump(dw, open("/mydata/results-offline/dw.pkl", "wb"))
    
if __name__ == "__main__":
    countStat("/mydata/output-offline/")
    countStat("/mydata/output-2x-offline/")
    countStat("/mydata/output-5x-offline/")
    countStat("/mydata/output-10x-offline/")