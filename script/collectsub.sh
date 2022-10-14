#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
# echo "${#ARRAY[@]}"
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
# echo ${NAME}
mkdir -p $2/$NAME

for f in 7
do
    for s in 50 100 500 1000
    do
        python3 ./algs/hierarchy-static.py -t $TRACE -o $2/$NAME -f ${f} -s ${s} -h 100000 -d 10000000 > $2/$NAME/f${f}-s${s}.txt
    done
done

