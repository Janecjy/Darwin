#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do
    ARRAY=(${TRACE//'/'/ })
    # echo "${#ARRAY[@]}"
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    # echo ${NAME}
    mkdir -p $2$NAME
    for f in 2 3 4 5 6 7
    do
        for s in 10 20 50 100 500 1000
        do
            python3 ./algs/hierarchy-static-results.py -t $TRACE -o $2$NAME -f ${f} -s ${s} -h 100000 -d 10000000 > $2$NAME/f${f}-s${s}.txt &
        done
    done
    wait
done
