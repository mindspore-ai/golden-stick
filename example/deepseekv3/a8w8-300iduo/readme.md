# DeepSeekR1网络SmoothQuant A8W8量化算法指南

本指南基于三机24卡，仅推理使用。

运行前请检查yaml配置中的tp并行数，和pp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 测试网络对话精度
命令如下，默认配置为三机24卡，local_rank_nums为当前机器的卡数如：8，master_ip多机时的主ip，rank_node为当前机器的node，使用的yaml文件为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径设置为/path/to/dst_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer.sh /path/to/mindformers local_rank_nums master_ip rank_node
```

## 2. 网络ceval数据集评测

命令如下，默认配置为三机24卡，需要传入ceval数据集路径，使用的yaml文件与第一步相同，为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径设置为/path/to/dst_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash eval_ceval.sh /path/to/mindformers /path/to/ceval_dataset_path
```

实测ceval(acc)结果为：89.45%

## 3. 网络gsm8k数据集评测

命令如下，默认配置为三机24卡，需要传入gsm8k数据集路径，使用的yaml文件与第二步相同，为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径设置为/path/to/dst_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash eval_gsm8k.sh /path/to/mindformers /path/to/ceval_dataset_path
```

实测ceval(acc)结果为：92.48%