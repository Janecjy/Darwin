#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do
    echo $2
    ARRAY=(${TRACE//'/'/ })
    # echo "${#ARRAY[@]}"
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    echo ${NAME}
    ./script/onlinesub.sh $TRACE $2 &
    python3 ./algs/hierarchy-online-3.py -t $TRACE -m /mydata/ -h 100000 -d 10000000 > $2/$NAME/online-3.out
done
wait
