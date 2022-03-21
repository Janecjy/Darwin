#!/bin/bash

COUNT=0
FILES="/home/janechen/cache/traces/volincrease/*"
for TRACE in $FILES
do
    ./script/collectfeaturesub.sh $TRACE &
    ((COUNT++))
    if [ $COUNT -eq 40 ]
        then
            wait
            COUNT=0
    fi
done
