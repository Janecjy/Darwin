#!/bin/bash

BASE_DIR="/scratch1/09498/janechen/mydata/"
for r in 1 2 5 10
do
    FILES=$BASE_DIR"tragen-traces-"$r"x/*"
    for TRACE in $FILES
    do
        NAME_EXT=$(basename "$TRACE")  # Extract the filename with extension
        NAME=${NAME_EXT%.*}
        regex='.*-[0-6]$'
        if [[ $NAME =~ $regex ]]
        then
            CORRECT_FEATURE_PATH=$BASE_DIR"tragen-features-"$r"x/"$NAME
            WRONG_FEATURE_PATH=$BASE_DIR"tragen-features-"$r"x/"$NAME$NAME
            echo "correct path: "$CORRECT_FEATURE_PATH
            echo "correct path file number: "$(ls $CORRECT_FEATURE_PATH | wc -l)
            echo "wrong path: "$WRONG_FEATURE_PATH
            echo "wrong path file number: "$(ls $WRONG_FEATURE_PATH | wc -l)
        fi
    done
done
