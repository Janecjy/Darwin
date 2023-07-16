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
            WRONG_FEATURE_PATH=$BASE_DIR"tragen-features-"$r"x/"$NAME"/"$NAME$NAME
            mv $WRONG_FEATURE_PATH/* $CORRECT_FEATURE_PATH/
            rmdir $WRONG_FEATURE_PATH
        fi
    done
done
