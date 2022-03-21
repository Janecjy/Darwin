#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[5]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
echo ${NAME}

python ./algs/hierarchy.py -t /home/janechen/cache/traces/feb3/$NAME.txt -f 2 -s 50 -h 100000 -d 10000000
python ./algs/hierarchy.py -t /home/janechen/cache/traces/feb3/$NAME.txt -f 2 -s 1000 -h 100000 -d 10000000
python ./algs/hierarchy.py -t /home/janechen/cache/traces/feb3/$NAME.txt -f 4 -s 50 -h 100000 -d 10000000
