#!/bin/bash

COUNT=0
FILES="/home/janechen/cache/traces/feb3-new/*"
for TRACE in $FILES
do
    ./script/collectsub.sh $TRACE &
    ((COUNT++))
    if [ $COUNT -eq 40 ]
        then
            wait
            COUNT=0
    fi
done

