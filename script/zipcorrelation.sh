#!/bin/bash

cd /mydata/correlations/
zip -r corr-$i.zip * &
wait