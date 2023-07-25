#!/bin/bash

BASE_DIR="/scratch2/09498/janechen/"
l=500000
for r in 1 5
do
    FILES="/scratch1/09498/janechen/mydata/tragen-traces-"$r"x/*"
    for c in 10 20
    do
        OUTPUT_DIR=$BASE_DIR"tragen-output-hillclimbing-c"$c"-"$r"x/"
        mkdir -p $OUTPUT_DIR
        for TRACE in $FILES
        do
            NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
            NAME=${NAME_EXT%.*}
            regex='.*-[7-9]$'
            if [[ $NAME =~ $regex ]]
            then
                echo "python3 algs/hillclimbing-continuous.py -t "$TRACE" -o "$OUTPUT_DIR" -h "$((100000 * r))" -d "$((10000000 * r))" -l "${l}" -c "${c}" > "$OUTPUT_DIR$NAME".txt" >> "./script/tacc_scripts/hillclimbing/hillclimbing_c"$c"_"$r"x_job"
            fi
        done
    done
done
