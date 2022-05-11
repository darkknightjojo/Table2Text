#!/bin/bash

echo "send data to es"

files=$(ls ./*.json)
i=1
progress_bar=''
length=$(ls -l | grep "^-" | wc -l)
length=`expr ${length} - 1`
echo ${length}
for f in ${files}
    do
        echo ${f} : ${i} / ${length} 
        if [ ${i} -eq 1 ]
        then
            curl -s -H "Content-Type: application/json" -XPOST "192.168.100.92:9200/aminer_authors/_bulk" --data-binary "@${f}"
        else
            # 不打印输出
            curl -H "Content-Type: application/json" -XPOST "192.168.100.92:9200/aminer_authors/_bulk" --data-binary "@${f}" >/dev/null 2>&1
        fi
        i=`expr $i + 1`
    done