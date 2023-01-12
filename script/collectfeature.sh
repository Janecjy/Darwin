#!/bin/bash

COUNT=0
FILES=$1'*'
i=$3
s=$4
for TRACE in $FILES
do  
    mkdir -p $2
    ./script/collectfeaturesub.sh $TRACE $2 $i $s &
    ((COUNT++))
    if [ $COUNT -eq 50 ]
        then
            wait
            COUNT=0
    fi
done
wait
