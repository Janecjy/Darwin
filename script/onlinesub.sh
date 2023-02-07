#!/bin/bash

#!/bin/bash

TRACE=$1
ARRAY=(${TRACE//'/'/ })
# echo "${#ARRAY[@]}"
FILENAME=${ARRAY[${#ARRAY[@]}-1]}
FILENAMEARR=(${FILENAME//./ })
NAME=${FILENAMEARR[0]}
mkdir -p $2$NAME
echo ${NAME}

python3 ./algs/hierarchy-online.py -t $TRACE -m /mydata/ -h 100000 -d 10000000 > $2$NAME/online.out
