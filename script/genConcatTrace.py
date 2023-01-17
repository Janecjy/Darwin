import os
import sys


dirPath = "/mydata/traces-online/"
outputName = "concat1.txt"

def main(trace_confs):
    id_start = 0
    id_max = 0
    t_start = 0
    t_max = 0
    
    outf = open(dirPath+outputName, "w")
    for tc in trace_confs:
        trace_name = "tc-0-1-"+tc+".txt"
        if os.path.exists(dirPath+trace_name):
            print(trace_name)
            with open(dirPath+trace_name) as inf:
                for line in inf:
                    t = line.split(',')[0]
                    id = line.split(',')[1]
                    new_t = t_start+int(t)
                    new_id = id_start+int(id)
                    new_line = str(new_t)+","+str(new_id)+","+line.split(',')[2]
                    outf.write(new_line)
                    if new_t > t_max:
                        t_max = new_t
                    if new_id > id_max:
                        id_max = new_id
        id_start = id_max+1
        t_start = t_max+1
        print(t_start, id_start)
            
    outf.close()

if __name__ == "__main__":
    trace_confs = sys.argv[1].split(',')
    main(trace_confs)