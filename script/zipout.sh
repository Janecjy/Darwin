#!/bin/bash

ID=$1

cd /mydata/output-offline/; 
zip -r out$ID.zip *
