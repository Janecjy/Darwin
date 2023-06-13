#!/bin/bash

cd /mydata
for i in 1 2 3 4 5
do
    mkdir -p node$i/traces
    mkdir -p node$i/output-offline
    mkdir -p node$i/features
    mkdir -p node$i/correlations
done

mv traces/tc-0-1-$1* node1/traces/
mv output-offline/tc-0-1-$1* node1/output-offline/
mv features/tc-0-1-$1* node1/features/
mv correlations/* node1/correlations/

mv traces/tc-0-1-$2* node2/traces/
mv output-offline/tc-0-1-$2* node2/output-offline/
mv features/tc-0-1-$2* node2/features/

mv traces/tc-0-1-$3* node3/traces/
mv output-offline/tc-0-1-$3* node3/output-offline/
mv features/tc-0-1-$3* node3/features/

mv traces/tc-0-1-$4* node4/traces/
mv output-offline/tc-0-1-$4* node4/output-offline/
mv features/tc-0-1-$4* node4/features/

mv traces/tc-0-1-$5* node5/traces/
mv output-offline/tc-0-1-$5* node5/output-offline/
mv features/tc-0-1-$5* node5/features/