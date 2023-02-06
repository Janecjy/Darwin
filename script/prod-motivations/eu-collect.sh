#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
# echo "${#ARRAY[@]}"
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[1]}
# echo ${NAME}
mkdir -p $2$NAME
COUNT=0

for f in 2 3 4 5 6 7
# for f in 5 6 7
do
    for s in 5 10 20 50 100 1000 5000 10000 20000
    do
        python3 ./algs/hierarchy-static-results-eu.py -t $TRACE -o $2$NAME -f ${f} -s ${s} -h 100000 -d 10000000 > $2$NAME/f${f}-s${s}.txt &
    done
done

