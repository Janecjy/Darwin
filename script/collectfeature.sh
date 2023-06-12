#!/bin/bash

COUNT=0
FILES=$1'*'
i=$3
s=$4
for TRACE in $FILES
do  
    ARRAY=(${TRACE//'/'/ })
    # echo "${#ARRAY[@]}"
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    if [[ ! -e $2$NAME/3M.pkl ]]; then
        mkdir -p $2
        ./script/collectfeaturesub.sh $TRACE $2 $i $s &
        ((COUNT++))
        if [ $COUNT -eq 30 ]
            then
                wait
                COUNT=0
        fi
    fi
done
wait
