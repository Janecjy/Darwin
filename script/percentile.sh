#!/bin/bash

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
    COUNT=0
    COUNT=0
    for f in 10 20 30 40 50 60 70 80 90
    do
        for s in 10 20 30 40 50 60 70 80 90
        do
            for l in 1000000 3000000
            do
                ((COUNT++))
                python3 algs/percentile.py  -t $TRACE -o $2$NAME -f ${f} -s ${s} -h 100000 -d 10000000 -l ${l} > $2$NAME/f${f}-s${s}-l${l}.txt &
                if [ $COUNT -eq 30 ]
                then
                    wait
                    COUNT=0
                fi
            done
        done
    done
done