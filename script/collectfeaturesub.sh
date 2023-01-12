#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}

source venv/bin/activate
mkdir -p $2$NAME
python ./algs/utils/traffic_model/extract_feature.py $TRACE $2$NAME $3 $4
