#!/bin/bash

for i in 1 2 3 4 5
do
    cd /mydata/node$i
    zip -r node$i.zip * &
done