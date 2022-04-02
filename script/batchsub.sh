#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[3]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
echo ${NAME}
mkdir -p /home/janechen/output/$NAME

python ./algs/hierarchy.py -t /home/janechen/volincrease/$NAME.txt -f 2 -s 0 -h 100000 -d 10000000 > /home/janechen/output/$NAME/f2-s0.txt
for f in 2 4 5 7
do
    for s in 50 100 200 500 1000
    do
        python ./algs/hierarchy.py -t /home/janechen/volincrease/$NAME.txt -f ${f} -s ${s} -h 100000 -d 10000000 > /home/janechen/output/$NAME/f${f}-s${s}.txt
    done
done
