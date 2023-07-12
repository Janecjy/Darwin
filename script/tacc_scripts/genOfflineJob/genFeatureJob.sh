#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
FEATURE_DIR="/scratch1/09498/janechen/mydata/tragen-features/"
for r in 1 2 5 10
do
    FILES=$BASE_DIR"tragen-traces-"$r"x/*"

    for TRACE in $FILES
    do
        NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
        NAME=${NAME_EXT%.*}
        regex='.*-[0-6]$'
        if [[ "$NAME" =~ $regex ]] 
        then
            # generate features
            mkdir -p $FEATURE_DIR$NAME
            echo "./script/collectfeaturesub.sh "$TRACE" "$FEATURE_DIR$NAME" 50000 150000" >> ./script/tacc_scripts/genFeatures/genFeatures_job
        fi
    done
done