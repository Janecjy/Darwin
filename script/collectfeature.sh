#!/bin/bash

COUNT=0
FILES=$1'*'
for TRACE in $FILES
do  
    for i in 50000 80000 90000 110000 120000 150000
    do
        for s in 50000 80000 90000 100000 110000 120000 150000
        do
            mkdir -p $2-$i-$s/
            ./script/collectfeaturesub.sh $TRACE $2-$i-$s/ $i $s &
            ((COUNT++))
            if [ $COUNT -eq 50 ]
                then
                    wait
                    COUNT=0
            fi
        done
    done
done
wait
