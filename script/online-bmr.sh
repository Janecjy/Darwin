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
    mkdir -p $2$NAME
    echo ${NAME}

    python3 ./algs/hierarchy-online-bmr.py -t $TRACE -m /mydata/ -h 100000 -d 10000000 > $2$NAME/online-bmr.out &
done
wait
