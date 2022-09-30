#!/bin/bash

for f in /mydata/traces/*
do
    RESULT=$(wc -l $f)
    ARRAY=(${RESULT//' '/ })
    LENGTH=${ARRAY[0]}
    FILENAME=${ARRAY[${#ARRAY[@]}-1]}
    echo $LENGTH $FILENAME
    if [ $LENGTH -lt 100000000 ]
    then
        rm $FILENAME
    fi
    if [ $LENGTH -gt 100000001 ]
    then
        head -n 100000001 $FILENAME > $FILENAME.head
        mv $FILENAME.head $FILENAME
    fi
done
