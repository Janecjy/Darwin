#!/bin/bash

COUNT=0
FILES="/home/janechen/cache/traces/test-set/*"
for TRACE in $FILES
do
    ./script/onlinesub.sh $TRACE &
    ((COUNT++))
    if [ $COUNT -eq 5 ]
        then
            wait
            COUNT=0
    fi
done
