#!/bin/bash

r=$1
HIDDEN=2
BASE_DIR="/scratch2/09498/janechen/"
for f0 in 2 3 4 5 6 7
do
    for s0 in $((10 * r)) $((20 * r)) $((50 * r)) $((100 * r)) $((500 * r)) $((1000 * r))
    do
        for f1 in 2 3 4 5 6 7
        do
            for s1 in $((10 * r)) $((20 * r)) $((50 * r)) $((100 * r)) $((500 * r)) $((1000 * r))
            do
                if [ ${f0} != ${f1} ] || [ ${s0} != ${s1} ]
                then
                    EXPERT0=f${f0}s${s0}
                    EXPERT1=f${f1}s${s1}
                    CORR_DIR=$BASE_DIR"tragen-correlations-"$r"x/"${EXPERT0}-${EXPERT1}
                    MODEL_DIR=$BASE_DIR"tragen-models-"$r"x/"${EXPERT0}-${EXPERT1}
                    mkdir -p $MODEL_DIR
                    echo "python3 algs/train.py "$HIDDEN" "${EXPERT0}" "${EXPERT1}" "$CORR_DIR" "$MODEL_DIR" > "$MODEL_DIR"/"$HIDDEN"-result.out" >> "./script/tacc_scripts/train/train_"$r"x_job"
                fi
            done
        done
    done
done
# done

