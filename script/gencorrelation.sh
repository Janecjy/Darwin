#!/bin/bash

COUNT=0
FILES=$1'*'
mkdir -p /mydata/correlations/
for TRACE in $FILES
do
    for f0 in 2 3 4 5 6 7
    do
        for s0 in 10 20 50 100 500 1000
        do
            for f1 in 2 3 4 5 6 7
            do
                for s1 in 10 20 50 100 500 1000
                do
                    if [ ${f0} != ${f1} ] || [ ${s0} != ${s1} ]
                    then
                        ((COUNT++))
                        EXPERT0=f${f0}s${s0}
                        EXPERT1=f${f1}s${s1}
                        mkdir -p /mydata/correlations/${EXPERT0}-${EXPERT1}
                        ARRAY=(${TRACE//'/'/ })
                        # echo "${#ARRAY[@]}"
                        FILENAME=${ARRAY[${#ARRAY[@]}-1]}
                        FILENAMEARR=(${FILENAME//./ })
                        NAME=${FILENAMEARR[0]}
                        if [[ -e /mydata/output-offline/$NAME/$EXPERT0-hits.txt ]] && [[ -e /mydata/output-offline/$NAME/$EXPERT1-hits.txt ]] && [[ $(wc -l < /mydata/output-offline/$NAME/$EXPERT0-hits.txt) -eq $(wc -l < /mydata/output-offline/$NAME/$EXPERT1-hits.txt) ]]
                        then
                            python3 algs/correlation_data_gen.py ${EXPERT0} ${EXPERT1} ${TRACE} > /mydata/correlations/${EXPERT0}-${EXPERT1}/$NAME.out &
                            if [ $COUNT -eq 30 ]
                            then
                                wait
                                COUNT=0
                            fi
                        fi
                    fi
                done
            done
        done
    done
done
wait

