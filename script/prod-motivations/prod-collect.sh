#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
# echo "${#ARRAY[@]}"
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
# echo ${NAME}
mkdir -p $2$NAME
COUNT=0

for f in 1 2 3 4 5 7
do
    for s in 10 20 50 1000 5000 10000
    do
        python3 ./algs/hierarchy-static-results-prod.py -t $TRACE -o $2$NAME -f ${f} -s ${s} -h 100000 -d 10000000 > $2$NAME/f${f}-s${s}.txt &
    done
done

