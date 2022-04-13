# Table2Text
毕设模型代码

## 预训练

##### 数据处理
将训练数据的前10000行复制到新文件
```
head -n 10000 data/wikibio/full/train_input.txt > data/wikibio/full/1k/train_input.txt
```
```
python3 script/create_json_data.py --src_data data/wikibio/full/1k/train_input.txt --output_data data/pretrain/input_1k.jsonl --task_type train --data_num 72831
```
##### 获取table embedding
```
# 需要先在pretrain/cfg里的配置文件配置输入数据路径
CUDA_VISIBLE_DEVICES=0 python3 core/pretrain/pretrain_main.py --embedding_file_path experiments/wikibio/pretrain/embeddings/train_table_embedding_1k --data_num 1000 --save_step 500
```

## 预处理数据
```
python3 main.py --preprocess --config cfg/preprocess.cfg --overwrite
```
## 训练：
```
python3 main.py --preprocess --config cfg/preprocess.cfg --overwrite
python3 main.py --train --config cfg/train_switch.cfg
```
## 测试：
```
python3 batch_translate.py --dataset wikibio --setname test --experiment switch5w --bsz 64 --bms 10 --start-step 5000 --gpu 0
python3 batch_compute_ngram_metrics.py --tables data/wikibio/full/test_tables.jl --references data/wikibio/test_output.txt --hypotheses experiments/switch5w/gens/test/
```

