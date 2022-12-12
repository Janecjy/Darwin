#!/bin/bash

COUNT=0
FILES=$1'*/'
mkdir -p /mydata/featurediff-req
source venv/bin/activate
for TRACE in $FILES
do
    if [ -f $TRACE"9M.pkl" ]; then
        python ./script/calfeaturediff.py $TRACE
    fi
done
