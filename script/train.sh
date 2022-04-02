#!/bin/bash

for f in 2 4 5 7
do
    for s in 50 1000
    do
        if [ $f != 4 ] || [ $s != 50 ]
        then
            mkdir -p ../cache/output/models/f4s50-f${f}s${s}
            python algs/importance_sampling.py 5 f4s50 f${f}s${s} > ../cache/output/models/f4s50-f${f}s${s}/result.out &
        fi
    done
done

