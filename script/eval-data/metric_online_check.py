import os
import pickle

input_dir = "/mydata/results-online/"

bmr = {}
dw = {}
ohr = {}

def addToMap(dst, src_f):
    sub_result = pickle.load(open(src_f, "rb"))
    dst.update(sub_result)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.startswith("bmr"):
            addToMap(bmr, root+"/"+file)
        if file.startswith("dw"):
            addToMap(dw, root+"/"+file)
        if file.startswith("ohr"):
            addToMap(ohr, root+"/"+file)
        # print(bmr)
        # print(dw)
        # print(ohr)
        # exit(0)

# assert(len(bmr.keys()) == 700), len(bmr.keys())
# assert(len(dw.keys()) == 700), len(dw.keys())
# assert(len(ohr.keys()) == 700), len(ohr.keys())
pickle.dump(bmr, open(input_dir+"bmr.pkl", "wb"))
pickle.dump(dw, open(input_dir+"dw.pkl", "wb"))
pickle.dump(ohr, open(input_dir+"ohr.pkl", "wb"))
