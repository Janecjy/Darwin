#!/bin/bash

BASE_DIR="/scratch2/09498/janechen/"
r=$1
OUTPUT_DIR=$BASE_DIR"tragen-output-offline-"$r"x/"
CORR_DIR=$BASE_DIR"tragen-correlations-"$r"x/"
FEATURE_DIR=$BASE_DIR"tragen-features-"$r"x/"
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
                    # if directory not empty
                    if [ ! "$(ls -A $CORR_DIR${EXPERT0}-${EXPERT1})" ]
                    then
                        mkdir -p $CORR_DIR${EXPERT0}-${EXPERT1}
                        # Use find command to get a list of directories under BASE_DIR
                        directories=$(find "$FEATURE_DIR" -type d -mindepth 1 -maxdepth 1)

                        # Iterate through each directory and print its name
                        for dir in $directories; do
                            NAME="$(basename "$dir")"
                            echo "python3 algs/correlation_data_gen_w_size.py "${EXPERT0}" "${EXPERT1}" "${NAME}" "$BASE_DIR" "$OUTPUT_DIR" "$CORR_DIR >> "./script/tacc_scripts/genCorr/genCorrWData_"$r"x_job"
                            fi
                        done
                    fi
                fi
            done
        done
    done
done
