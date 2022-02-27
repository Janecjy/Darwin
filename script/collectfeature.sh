#!/bin/bash

COUNT=0
FILES="/home/janechen/cache/traces/feb3/*"
for TRACE in $FILES
do
    ./script/collectfeaturesub.sh $TRACE &
    ((COUNT++))
    if [ $COUNT -eq 20 ]
        then
            wait
            COUNT=0
    fi
done
