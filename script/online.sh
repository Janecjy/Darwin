#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do
    echo $2
    ./script/onlinesub.sh $TRACE $2 &
    ((COUNT++))
    if [ $COUNT -eq 30 ]
        then
            wait
            COUNT=0
    fi
done
wait
