#!/bin/bash

EXPERT=$1
mkdir -p /mydata/experts/$EXPERT

for TRACE in /mydata/output/*
do
    ARRAY=(${TRACE//'/'/ })
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    echo $NAME

    cp $TRACE/$EXPERT-hits.pkl /mydata/experts/$EXPERT/$NAME.pkl
done