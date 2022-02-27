#!/bin/bash

COUNT=0
FILES="/home/janechen/cache/traces/feb3-new/*"
for TRACE in $FILES
do
    ./script/batchsub.sh $TRACE &
    ((COUNT++))
    if [ $COUNT -eq 20 ]
        then
            wait
            COUNT=0
    fi
done
# python ./algs/draw.py /home/janechen/cache/output
