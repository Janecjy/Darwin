import pickle
import os
# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

plt.rcParams.update({'font.size': 15})
input_dir = "/mydata/output-production/eu-subtraces/"
output_dir = "/mydata/output-production/eu-subtraces/figs/"
for trace in os.listdir(input_dir):
    
    if trace != "figs":
        # draw ohr
        ohr_result_path = input_dir+trace+"/ohr-result.pkl"

        ohr_res = pickle.load(open(ohr_result_path, "rb"))
        # print(ohr_res[trace])
        fs = []
        ss = []
        ohrs = []
        for k, v in ohr_res[trace].items():
            f = int(k[1])
            s = int(k[3:])
            # print(f, s)
            if s in [100, 1000, 2000, 4000, 20000] and f in [1, 2, 3, 4, 5]:
                fs.append(f)
                ss.append(s)
                ohrs.append(v)
        assert len(fs) == len(ss) and len(fs) == len(ohrs), str(len(fs))+" "+str(len(ss))+" "+str(len(ohrs))

        df = pd.DataFrame({"Frequency Threshold": fs, "Size Threshold (KB)": ss, "ohr": ohrs})
        
        result = df.pivot(index="Size Threshold (KB)", columns="Frequency Threshold", values='ohr')
        print(result)
        
        plt.figure()

        ax = sns.heatmap( result ,  annot=True, linewidth = 0.5 , cmap = 'coolwarm', annot_kws={'size': 15}, fmt='.2f', cbar=False )
        ax.invert_yaxis()
        plt.savefig(output_dir+trace+"-ohr-heat.png",bbox_inches='tight')
        
        
        # draw disk writes
        dw_result_path = input_dir+trace+"/dw-result.pkl"

        dw_res = pickle.load(open(dw_result_path, "rb"))
        # print(ohr_res[trace])
        fs = []
        ss = []
        dws = []
        for k, v in dw_res[trace].items():
            f = int(k[1])
            s = int(k[3:])
            # print(f, s)
            if s in [100, 1000, 2000, 4000, 20000] and f in [1, 2, 3, 4, 5]:
                fs.append(f)
                ss.append(s)
                dws.append(int(v))
        assert len(fs) == len(ss) and len(fs) == len(dws), str(len(fs))+" "+str(len(ss))+" "+str(len(ohrs))

        df = pd.DataFrame({"Frequency Threshold": fs, "Size Threshold (KB)": ss, "dw": dws})
        
        result = df.pivot(index="Size Threshold (KB)", columns="Frequency Threshold", values='dw')
        print(result)
        
        plt.figure()

        ax = sns.heatmap( result ,  annot=True, linewidth = 0.5 , cmap = 'coolwarm', annot_kws={'size': 15}, fmt='.1e', cbar=False )
        ax.invert_yaxis()
        plt.savefig(output_dir+trace+"-dw-heat.png",bbox_inches='tight')
