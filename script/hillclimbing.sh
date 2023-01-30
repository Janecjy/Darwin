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
    for l in 100000 500000 1000000 5000000 10000000 
    do
        ((COUNT++))
        python3 algs/hillclimbing.py  -t $TRACE -o $2$NAME -h 100000 -d 10000000 -l ${l} > $2$NAME/l${l}.txt &
        # if [ $COUNT -eq 30 ]
        # then
        #     wait
        #     COUNT=0
        # fi
    done
done