# DeepSeekR1网络Flash Attention 3(FA3) 量化算法指南

Flash Attention 3(FA3)校准算法现属于demo特性。

本指南基于单机16卡，如果使用双机16卡，请将msrun命令替换为双机16卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下，默认配置为单机16卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_calibrate.yaml，需要修改yaml文件的load_checkpoint、vocab_file、tokenizer_file参数， 默认生成fa3校准参数位于当前目录fa3_params文件夹下：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate.sh /path/to/mindformers
```
