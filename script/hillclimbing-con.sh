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
    l=500000
    for c in 1 10 20 50 100 
    do
        ((COUNT++))
        python3 algs/hillclimbing-continuous.py  -t $TRACE -o $2$NAME -h 100000 -d 10000000 -l ${l} -c ${c} > $2$NAME/l${l}-c${c}.txt &
        # if [ $COUNT -eq 30 ]
        # then
        #     wait
        #     COUNT=0
        # fi
    done
done
wait