首先需要将aminer的数据进行分片，否则太大无法上载到es中,需要将数据划分为50M以下的文件  
这里使用linux的split命令进行分片，为了保证数据的完整性，以行数作为分割条件
```
split -l 60000 source_file prefix
```
分割完成后，需要处理成es可识别的格式，调用create_es_data命令进行处理
```
python3 create_es_data.py
```
然后使用脚本upload_data_to_es.sh将数据上载到es中
```
sh upload_data_to_es.sh
```