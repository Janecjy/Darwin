#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
FILES=$BASE_DIR"traces/*"
OUTPUT_DIR=$BASE_DIR"output-offline/"
r=1
for TRACE in $FILES
do
    NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
    NAME=${NAME_EXT%.*}
    mkdir -p $OUTPUT_DIR$NAME
    for f in 2 3 4 5 6 7
    do
        for s in $((30 * r)) $((40 * r)) $((60 * r)) $((80 * r)) $((150 * r)) $((200 * r)) $((300 * r)) $((400 * r)) $((600 * r)) $((800 * r))
        do
            echo "python3 ./algs/hierarchy-static-results.py -t "$TRACE" -o "$OUTPUT_DIR$NAME" -f "${f}" -s "${s}" -h "$((100000 * r))" -d "$((10000000 * r))" > "$OUTPUT_DIR$NAME"/f"${f}"-s"${s}".txt" &
        done
    done
done
