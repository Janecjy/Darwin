#!/bin/bash

COUNT=0
FILES="/mydata/models/*"
for MODEL in $FILES
do
    rm -f $MODEL/model-h2-*.ckpt 
    tail -n 43202 $MODEL/2-result.out > $MODEL/2.result
    # break
done
# python ./algs/draw.py /home/janechen/cache/output
