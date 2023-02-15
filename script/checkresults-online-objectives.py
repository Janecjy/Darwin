import re
import os
import pickle


def countStat(dirPath):
    ohr = {}
    bmr = {}
    dw = {}
    current_root = None
    
    for root, dirs, files in os.walk(dirPath):
        
        for file in files:
            file_res = []
            if file.endswith(".txt"):
                trace = root.split('/')[-1]

                if trace not in ohr.keys():
                    ohr[trace] = {}
                    bmr[trace] = {}
                    dw[trace] = {}
                
                f = file.split('-')[0].replace("f", "")
                s = file.split('-')[1].split('.')[0].replace("s", "")

                for line in open(os.path.join(root, file), "r"):
                    
                    val2 = re.findall('final hoc hit: [\d]+[.]?[\d]*%, hoc byte miss: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                    
                    for sentence in val2:
                        exp = sentence.split(',')
                        exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                        file_res.append(exp)
                if len(file_res)>0:
                    ohr[trace]["f"+f+"s"+s] = [x[0] for x in file_res][-1]
                    bmr[trace]["f"+f+"s"+s] = [x[1] for x in file_res][-1]
                    dw[trace]["f"+f+"s"+s] = [x[5] for x in file_res][-1]
            
            if file.endswith('ohr-dw.out'):
                trace = root.split('/')[-1]

                if trace not in ohr.keys():
                    ohr[trace] = {}
                    bmr[trace] = {}
                    dw[trace] = {}

                for line in open(os.path.join(root, file), "r"):
                    
                    val2 = re.findall('hoc hit: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                    
                    for sentence in val2:
                        exp = sentence.split(',')
                        exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                        file_res.append(exp)
                if len(file_res)>0:
                    ohr[trace]["online"] = [x[0] for x in file_res][-1]
                    dw[trace]["online"] = [x[4] for x in file_res][-1]
                    
            if file.endswith('bmr.out'):
                trace = root.split('/')[-1]

                if trace not in ohr.keys():
                    ohr[trace] = {}
                    bmr[trace] = {}
                    dw[trace] = {}

                for line in open(os.path.join(root, file), "r"):
                    
                    val2 = re.findall('hoc hit: [\d]+[.]?[\d]*%, hoc bmr: [\d]+[.]?[\d]*%, hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                    
                    for sentence in val2:
                        exp = sentence.split(',')
                        exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                        file_res.append(exp)
                if len(file_res)>0:
                    bmr[trace]["online"] = [x[1] for x in file_res][-1]
            

    pickle.dump(ohr, open("/mydata/results-online/ohr.pkl", "wb"))
    pickle.dump(bmr, open("/mydata/results-online/bmr.pkl", "wb"))
    pickle.dump(dw, open("/mydata/results-online/dw.pkl", "wb"))
    
if __name__ == "__main__":
    countStat("/mydata/output-offline-100M/")