#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
for r in 2 5 10
do
    OFFLINE_DIR=$BASE_DIR"output-offline-"$r"x/"
    OUTPUT_DIR=$BASE_DIR"correlations-"$r"x/"
    mkdir -p $OUTPUT_DIR
    FILES=$BASE_DIR"traces-"$r"x/*"
    for TRACE in $FILES
    do
        NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
        NAME=${NAME_EXT%.*}
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
                            EXPERT0=f${f0}s${s0}
                            EXPERT1=f${f1}s${s1}
                            mkdir -p $OUTPUT_DIR${EXPERT0}-${EXPERT1}
                            regex='.*-[0-6]$'
                            if [[ -e $OFFLINE_DIR$NAME/$EXPERT0-hits.txt ]] && [[ -e $OFFLINE_DIR$NAME/$EXPERT1-hits.txt ]] && [[ ! -e $OUTPUT_DIR${EXPERT0}-${EXPERT1}/$NAME-input.pkl ]]
                            then
                                echo "python3 algs/correlation_data_gen_w_size.py "${EXPERT0}" "${EXPERT1}" "${TRACE}" "$BASE_DIR" "$OFFLINE_DIR" "$OUTPUT_DIR" > "$OUTPUT_DIR${EXPERT0}"-"${EXPERT1}"/"$NAME".out"
                            fi
                        fi
                    done
                done
            done
        done
    done
done