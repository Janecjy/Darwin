#!/bin/bash

COUNT=0
FILES=$1'*/'
mkdir -p $TRACE-output
source venv/bin/activate
for TRACE in $FILES
do
    if [ -f $TRACE"/9M.pkl" ]; then
        python ./script/outputfeature.py $TRACE/
    fi
done
