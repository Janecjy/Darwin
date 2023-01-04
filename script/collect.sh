#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do
    ./script/collectsub.sh $TRACE $2
done
wait
