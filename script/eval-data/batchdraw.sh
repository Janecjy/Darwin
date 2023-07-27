#!/bin/bash

# for i in 1 to 20
for i in {1..20}
do
    python3 draw-ensemble-ohr-1x.py 1 > $i.txt
    python3 draw-ensemble-ohr-1x.py 5 >> $i.txt
    mv baseline-improvement-percentage-1x.png $i-1x.png
    mv baseline-improvement-percentage-5x.png $i-5x.png
done
zip figures.zip *.png