#!/bin/bash

COUNT=0
FILES=$1'*'
RATIO=$3
COUNTMAX=$4
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
        for s in $((10 * RATIO)) $((20 * RATIO)) $((50 * RATIO)) $((100 * RATIO)) $((500 * RATIO)) $((1000 * RATIO))
        do
            python3 ./algs/hierarchy-static-results.py -t $TRACE -o $2$NAME -f ${f} -s ${s} -h $((100000 * RATIO)) -d 10000000 > $2$NAME/f${f}-s${s}.txt &
            ((COUNT++))
            if [ $COUNT -eq $COUNTMAX ]
                then
                    wait
                    COUNT=0
            fi
        done
    done
    # wait
done
wait
