#!/bin/bash

COUNT=0
DEVICE="cuda"
# for HIDDEN in 2 3 4 5 6 7 8 9
# do
HIDDEN=2
for f0 in 7
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
                    ./script/trainsub.sh $HIDDEN f${f0}s${s0} f${f1}s${s1} $DEVICE &
                    
                    # if [ $COUNT -eq 8 ]
                    # then
                    #     DEVICE="cuda:1"
                    # fi
                    # if [ $COUNT -eq 16 ]
                    # then
                    #     DEVICE="cuda:2"
                    # fi
                    if [ $COUNT -eq 24 ]
                    then
                        wait
                        COUNT=0
                        # DEVICE="cuda:0"
                    fi
                fi
            done
        done
    done
done
# done

