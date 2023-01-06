#!/bin/bash

COUNT=0
DEVICE="cuda"
HIDDEN=2
for f0 in 2 3 4 5 6 7
do
    for s0 in 10 20 50 100 500 1000
    do
        for f1 in 2 3 4 5 6 7
        do
            for s1 in 10 20 50 100 500 1000
            do
                if [ ${f0} != ${f1} ] || [ ${s0} != ${s1} ]
                then
                    ((COUNT++))
                    python algs/ood_test.py $HIDDEN f${f0}s${s0} f${f1}s${s1} $DEVICE > /mydata/models/f${f0}s${s0}-f${f1}s${s1}/$HIDDEN-ood-result.out &
                    if [ $COUNT -eq 24 ]
                    then
                        wait
                        COUNT=0
                    fi
                fi
            done
        done
    done
done
