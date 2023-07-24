#!/bin/bash

r=1
BASE_DIR="/scratch2/09498/janechen/"
FILES="/scratch1/09498/janechen/mydata/tragen-traces-"$r"x/*"
OUTPUT_DIR=$BASE_DIR"tragen-output-online-"$r"x/"
mkdir -p $OUTPUT_DIR
for TRACE in $FILES
do
    NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
    NAME=${NAME_EXT%.*}
    regex='.*-[7-9]$'
    if [[ $NAME =~ $regex ]]
    then
        echo "python3 ./algs/hierarchy-online.py -t "$TRACE" -m "$BASE_DIR"tragen-models-"$r"x -h "$((100000 * r))" -d "$((10000000 * r))" > "$OUTPUT_DIR$NAME".txt" > ./script/tacc_scripts/genOnline/genOnline1x_job
    fi
done
