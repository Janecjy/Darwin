#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do
    ./script/collectsub.sh $TRACE $2 &
    ((COUNT++))
    if [ $COUNT -eq  ]
        then
            wait
            COUNT=0
    fi
done
wait
