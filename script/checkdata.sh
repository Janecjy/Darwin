#!/bin/bash

FILES="/home/janechen/cache/output/features/*"
for TRACE in $FILES
do
    ARRAY=(${TRACE//'/'/ })
    FILENAME=${ARRAY[5]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    
    [ ! -d "/home/janechen/cache/output/"$NAME ] && echo "Directory "$NAME" DOES NOT exists."

done
