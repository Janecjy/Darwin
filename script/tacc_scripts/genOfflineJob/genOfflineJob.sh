#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
for r in 2 5 10
do
    FILES=$BASE_DIR"traces-"$r"x/*"
    OUTPUT_DIR=$BASE_DIR"output-offline-"$r"x/"
    for TRACE in $FILES
    do
        NAME_EXT=$(basename "$f")  # Extract the filename with extension
        NAME=${NAME_EXT%.*}
        mkdir -p $OUTPUT_DIR$NAME
        for f in 2 3 4 5 6 7
        do
            for s in $((10 * r)) $((20 * r)) $((50 * r)) $((100 * r)) $((500 * r)) $((1000 * r))
            do
                echo "python3 ./algs/hierarchy-static-results.py -t "$TRACE" -o "$OUTPUT_DIR$NAME" -f "${f}" -s "${s}" -h "$((100000 * r))" -d "$((10000000 * r))" > "$OUTPUT_DIR$NAME"/f"${f}"-s"${s}".txt" &
            done
        done
    done
done
