#!/bin/bash

COUNT=0
FILES=$1'*/'
rm -rf $2
mkdir -p $2
for TRACE in $FILES
do
    ARRAY=(${TRACE//'/'/ })
    # echo "${#ARRAY[@]}"
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    mv $TRACE"3M.pkl" $2$NAME.pkl
done
