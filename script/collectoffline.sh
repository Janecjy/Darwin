#!/bin/bash

EXPERT=$1
INPUT=$2
OUTPUT=$3
mkdir -p $3$EXPERT

for TRACE in $2*
do
    ARRAY=(${TRACE//'/'/ })
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    echo $NAME

    cp $TRACE/$EXPERT-hits.pkl $3$EXPERT/$NAME.pkl
done