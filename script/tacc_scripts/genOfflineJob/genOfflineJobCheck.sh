#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
r=$1
FILES=$BASE_DIR"tragen-traces-"$r"x/*"
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
                        for TRACE in $FILES
                        do
                            NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
                            NAME=${NAME_EXT%.*}
                            regex='.*-[0-6]$'
                            if [[ "$NAME" =~ $regex ]] 
                            then
                                # if [[ -e $OUTPUT_DIR$NAME/$EXPERT0-hits.txt ]] && [[ -e $OUTPUT_DIR$NAME/$EXPERT1-hits.txt ]] # && [[ $(wc -l < $OUTPUT_DIR$NAME/$EXPERT0-hits.txt) -eq 99000000 ]] && [[ $(wc -l < $OUTPUT_DIR$NAME/$EXPERT1-hits.txt) -eq 99000000 ]]
                                # then
                                # generate correlation with data
                                echo "python3 algs/correlation_data_gen_w_size.py "${EXPERT0}" "${EXPERT1}" "${TRACE}" "$BASE_DIR" "$OUTPUT_DIR" "$CORR_DIR >> "./script/tacc_scripts/genCorr/genCorrWData_"$r"x_job"
                                # else
                                #     # generate correlation without data
                                #     echo "python3 algs/correlation_gen_per_pair.py -t "${TRACE}" -i "${FEATURE_DIR}" -c "${CORR_DIR}" -f "${f0}" -s "${s0}" -g "${f1}" -u "${s1}" -h "$((100000 * r))" -d "$((10000000 * r)) >> "./script/tacc_scripts/genCorr/genCorrWoData_"$r"x_job"
                                # fi
                            fi
                        done
                    fi
                fi
            done
        done
    done
done
