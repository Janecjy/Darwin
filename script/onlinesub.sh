#!/bin/bash

#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
# echo "${#ARRAY[@]}"
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}

TRACE=$1
ARRAY=(${TRACE//'/'/ })
FILENAME=${ARRAY[5]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
echo $2/$NAME.out

python3 ./algs/hierarchy-online.py -t $TRACE -m /mydata/models/ -h 100000 -d 10000000 > $2/$NAME.out
