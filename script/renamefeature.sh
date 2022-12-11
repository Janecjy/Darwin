#!/bin/bash

COUNT=0
FILES='/mydata/features/-50000-150000/*/'
rm -rf "/mydata/final-features/"
mkdir -p "/mydata/final-features/"
for TRACE in $FILES
do
    ARRAY=(${TRACE//'/'/ })
    # echo "${#ARRAY[@]}"
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    mv $TRACE"3M.pkl" /mydata/final-features/$NAME.pkl
done
