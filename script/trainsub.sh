#!/bin/bash

HIDDEN=$1
EXPERT0=$2
EXPERT1=$3
DEVICE=$4

mkdir -p /mydata/models/${EXPERT0}-${EXPERT1}
python algs/importance_sampling.py $HIDDEN ${EXPERT0} ${EXPERT1} ${DEVICE} > /mydata/models/${EXPERT0}-${EXPERT1}/$HIDDEN-result.out
