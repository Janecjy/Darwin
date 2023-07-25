#!/bin/bash

BASE_DIR="/scratch2/09498/janechen/"
for r in 1 5
do
    FILES="/scratch1/09498/janechen/mydata/tragen-traces-"$r"x/*"
    OUTPUT_DIR=$BASE_DIR"tragen-output-percentile-"$r"x/"
    mkdir -p $OUTPUT_DIR
    for TRACE in $FILES
    do
        NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
        NAME=${NAME_EXT%.*}
        regex='.*-[7-9]$'
        if [[ $NAME =~ $regex ]]
        then
            echo "python3 algs/percentile.py -t "$TRACE" -o "$OUTPUT_DIR$NAME" -f 60 -s 90 -h "$((100000 * r))" -d "$((10000000 * r))" -l 100000 > "$OUTPUT_DIR$NAME".txt" >> "./script/tacc_scripts/runPercentile/percentile_"$r"x_job"
        fi
    done
done