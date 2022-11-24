#!/bin/bash

COUNT=0
FILES=$1'*/'
mkdir -p /mydata/featurediff
for TRACE in $FILES
do
    if [ -f $TRACE"9M.pkl" ]; then
        python calfeaturediff.py $TRACE
    fi
done
