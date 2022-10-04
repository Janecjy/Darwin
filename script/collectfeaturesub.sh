#!/bin/bash

TRACE=$1
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
# echo ${NAME}

python ./algs/utils/traffic_model/extract_feature.py $TRACE $2$NAME
