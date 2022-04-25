#!/bin/bash

FILES="/home/janechen/cache/traces/feb3/*"
for TRACE in $FILES
do
    ARRAY=(${TRACE//'/'/ })
    FILENAME=${ARRAY[5]}
    FILENAMEARR=(${FILENAME//./ })
    NAME=${FILENAMEARR[0]}
    
    [ ! -f "/home/janechen/cache/output/features/"$NAME".pkl" ] && echo "Feature file "$NAME" DOES NOT exists."

    [ ! -d "/home/janechen/cache/output/"$NAME ] && echo "Directory "$NAME" DOES NOT exists."

    [ ! -f "/home/janechen/cache/output/"$NAME"/f5s500-hits.pkl" ] && echo $NAME" hits data DOES NOT exists."

done
