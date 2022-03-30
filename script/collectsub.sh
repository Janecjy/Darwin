#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[5]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
echo ${NAME}

for f in 2 4 5 7
do
    for s in 50 100 200 500 1000
    do
        python ./algs/hierarchy.py -t /home/janechen/cache/traces/feb3-new/$NAME.txt -f ${f} -s ${s} -h 100000 -d 10000000 > /home/janechen/cache/output/$NAME/f${f}-s${s}.txt
    done
done

