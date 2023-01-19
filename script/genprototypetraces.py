import csv

dirPath = "/mydata/traces-online/"
inputName = "concat1"
with open(dirPath+inputName+".txt") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    client_f = open(dirPath+inputName+"-client.txt", "w")
    origin_f = open(dirPath+inputName+"-origin.txt", "w")
    for row in csv_reader:
        t = int(row[0])
        id = int(row[1])
        size = int(row[2])*1024
        client_f.write(str(t)+" "+str(id)+" "+str(size)+"\n")
        origin_f.write(str(id)+" "+str(size)+"\n")
client_f.close()
origin_f.close()