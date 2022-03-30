#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[5]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
# echo ${NAME}

python ./algs/utils/traffic_model/extract_feature.py $NAME
