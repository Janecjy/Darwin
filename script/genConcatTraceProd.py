import os
import sys
import random

# total length 8M
dirPath = "/mydata/traces-production/"
outputName = "concat1.txt"
traceLength = 8000000

def getDownloadMax():
    id_max = 0
    with open(dirPath+"download.txt") as inf:
        for line in inf:
            id = int(line.split()[1])
            if id > id_max:
                id_max = id
    print(id_max)
    
def main(trace_confs):
    id_start = 0
    id_max = 0
    id_off = 322962
    t_start = 0
    t_max = 0
    t = 1495781788
    
    outf = open(dirPath+outputName, "w")
    for tc in trace_confs:
        download_rate = float(tc.split(":")[0])/(float(tc.split(":")[0])+float(tc.split(":")[1].replace("\n", "")))
        # print(tc.split(":")[0])
        print(download_rate)
        inf_0_name = "download.txt"
        inf_1_name = "image.txt"
        inf_0_linenum = inf_1_linenum =  0
        with open(dirPath+inf_0_name) as inf_0:
            inf_0_contents = [next(inf_0) for x in range(traceLength)]
        with open(dirPath+inf_1_name) as inf_1:
            inf_1_contents = [next(inf_1) for x in range(traceLength)]
        for i in range(traceLength):
            if random.uniform(0, 1) <= download_rate:
                line = inf_0_contents[inf_0_linenum]
                t = int(line.split(' ')[0])+t_start
                id = int(line.split(" ")[1])+id_start
                outf.write(str(t)+','+str(id)+','+line.split(" ")[2])
                inf_0_linenum += 1
            else:
                line = inf_1_contents[inf_1_linenum]
                id = int(line.split(" ")[1])+id_start+id_off
                outf.write(str(t)+','+str(id)+','+line.split(" ")[2])
                inf_1_linenum += 1
            if id > id_max:
                id_max = id
            if t > t_max:
                t_max = id
        t_start = t_max + 1
        id_start = id_max + 1
            
    outf.close()

if __name__ == "__main__":
    # getDownloadMax()
    trace_confs = sys.argv[1].split(',')
    main(trace_confs)