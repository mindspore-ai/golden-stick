# DeepSeekR1网络OmniQuant A8W8量化算法指南

本指南基于单机16卡，如果使用双机16卡，请将msrun命令替换为双机16卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下，默认配置为单机16卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_calibrate.yaml，需要修改yaml文件的load_checkpoint、vocab_file、tokenizer_file参数，确保output目录下有足够的空间存放量化后的权重，约710GB：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate.sh /path/to/mindformers
```

## 2. 测试量化网络对话精度

命令如下，默认配置为单机16卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为./output/DeepSeekR1-W8A8_omniquant_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer.sh /path/to/mindformers
```

## 3. 量化网络ceval数据集评测

命令如下，默认配置为单机16卡，需要传入ceval数据集路径，使用的yaml文件与第二步相同，为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为默认路径./output/DeepSeekR1-W8A8_omniquant_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash eval_ceval.sh /path/to/mindformers /path/to/ceval_dataset_path
```

实测ceval(acc)结果为：88.93%
