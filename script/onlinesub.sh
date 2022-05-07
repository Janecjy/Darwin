#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[5]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
echo ${NAME}

python ./algs/hierarchy-online.py -t /home/janechen/cache/traces/test-set/$NAME.txt -h 100000 -d 10000000 > /home/janechen/MultiExpertHOCAdmission/$NAME.out
