# Table2Text
毕设模型代码

## 预训练
##### 数据处理
```
python3 data/create_json_data.py --train_data data/wikibio/full/1k/train_input.txt --output_data data/pretrain/input_1k.jsonl
```
##### 获取table embedding
```
python3 core/pretrain/pretrain_main.py
```

## 预处理数据
```
python3 main.py --preprocess --config cfg/preprocess.cfg --overwrite
```
## 训练：
```
python3 main.py --preprocess --config preprocess.cfg --overwrite
python3 main.py --train --config train_switch.cfg
```
## 测试：
```
python3 batch_translate.py --dataset wikibio --setname test --experiment switch5w --bsz 64 --bms 10 --start-step 5000 --gpu 0
python3 batch_compute_ngram_metrics.py --tables data/wikibio/full/test_tables.jl --references data/wikibio/test_output.txt --hypotheses experiments/switch5w/gens/test/
```
