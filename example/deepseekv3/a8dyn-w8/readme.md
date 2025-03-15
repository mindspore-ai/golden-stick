# DeepSeekR1网络A8DYN-W8量化算法指南

本指南基于单机16卡，如果使用双机16卡，请将msrun命令替换为双机16卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下，默认配置为单机16卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b.yaml，需要修改yaml文件的load_checkpoint、vocab_file、tokenizer_file参数，生成的权重路径为./output/DeepSeekR1_a8dynw8_safetensors：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash a8dynw8_calibrate_single.sh /path/to/mindformers
```

也可以根据需要传入work_num，和yaml文件路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash a8dynw8_calibrate_single.sh /path/to/mindformers worker_num /path/to/yaml
```

## 2. 测试图模式下的量化网络对话精度

命令如下，默认配置为单机16卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_a8dynw8.yaml，load_checkpoint参数路径已设置为./output/DeepSeekR1_a8dynw8_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash a8dynw8_quant_infer_single.sh /path/to/mindformers
```

也可以根据需要传入work_num，和yaml文件路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer_single.sh /path/to/mindformers worker_num /path/to/yaml
```

## 3. 测试图模式下的量化网络ceval数据集精度

命令如下，默认配置为单机16卡，需要传入ceval数据集路径，使用的yaml文件与第二步相同，为当前目录下的predict_deepseek_r1_671b_a8dynw8.yaml，load_checkpoint参数路径已设置为./output/DeepSeekR1_a8dynw8_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash a8dynw8_eval_ceval_single.sh /path/to/mindformers /path/to/ceval_dataset_path
```

也可以根据需要传入work_num，和yaml文件路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash a8dynw8_eval_ceval_single.sh /path/to/mindformers /path/to/ceval_dataset_path worker_num /path/to/yaml
```
