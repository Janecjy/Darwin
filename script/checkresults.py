import re
import os

def confSort(keys):
    return sorted(keys, key=lambda element: list(int(x.replace('f', '')) for x in element.split('s')[:]))

def countStat(dirPath):
    hr = {}
    current_root = None
    
    for root, dirs, files in os.walk(dirPath):
        
        for file in files:
            
            if not file.endswith(".txt"):
                continue
            
            if current_root == None:
                current_root = root
            if current_root != root:
                hr_max = max(list([hr[x][0] for x in hr.keys()]))
                
                best_set = []
                for x in hr.keys():
                    if hr_max - hr[x][0] < 1:
                        best_set.append(x)
                        
                print(current_root.split('/')[-1]+' '+' '.join(confSort(best_set)))
                hr = {}
                current_root = root
                
            file_res = []

            for line in open(os.path.join(root, file), "r"):
                val1 = re.findall('freq: [\d]*, size: [\d]*',line)

                for sentence in val1:
                    sentence = sentence.split(',')
                    f = (sentence[0].split(':')[1].replace(" ", ""))
                    s = (sentence[1].split(':')[1].replace(" ", ""))
                
                val2 = re.findall('hr: [\d]+[.]?[\d]*%, bmr: [\d]+[.]?[\d]*%, disk read: [\d]+[.]?[\d]*, disk write: [\d]+[.]?[\d]*',line)
                
                for sentence in val2:
                    exp = sentence.split(',')
                    exp = [float(x.replace(" ", "").replace("%", "").split(':')[1]) for x in exp]
                    file_res.append(exp)

            if file_res:
                hr['f'+f+'s'+s] = [x[0] for x in file_res]
                
    hr_max = max(list([hr[x][0] for x in hr.keys()]))
        
    best_set = []
    for x in hr.keys():
        if hr_max - hr[x][0] < 1:
            best_set.append(x)
            
    print(root.split('/')[-1]+' '+' '.join(confSort(best_set)))
    
if __name__ == "__main__":
    countStat("/mydata/output/")