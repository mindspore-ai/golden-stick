# DeepSeekR1网络A8W4量化算法指南

本指南基于单机8卡，如果使用单机8卡，请将msrun命令替换为单机8卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下，默认配置为单机8卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_calibrate.yaml，需要修改yaml文件的load_checkpoint、vocab_file、tokenizer_file参数，确保output目录下有足够的空间存放量化后的权重，约361GB：

量化算法逐层配置不同策略：

- MLA: 静态A8W8量化

- MLP: 动态A8W8量化

- MOE-路由专家：**动态A8W4量化**

- MOE-共享专家: 动态A8W8量化

可通过如下命令一键拉起进程，进行A8W4算法校准：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate.sh /path/to/mindformers 8
```

## 2. 测试量化网络对话精度

命令如下，默认配置为单机8卡，且使用的yaml文件为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为./output/DeepSeekR1_a8w4_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer.sh /path/to/mindformers 8
```

## 3. 量化网络ceval数据集评测

命令如下，默认配置为单机8卡，需要传入ceval数据集路径，使用的yaml文件与第二步相同，为当前目录下的predict_deepseek_r1_671b_qinfer.yaml，load_checkpoint参数路径已设置为默认路径./output/DeepSeekR1_a8w4_safetensors，需要修改yaml文件的vocab_file、tokenizer_file参数：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash eval_ceval.sh /path/to/mindformers /path/to/ceval_dataset_path 8
```

注: gptq量化时需要进行cholesky分解，mindspore不支持在Ascend环境进行cholesky分解，因此需要安装鲲鹏数学库kml来进行cholesky分解计算。

# 鲲鹏数学库的安装说明

请参考[gptq量化README.md](https://gitee.com/mindspore/golden-stick/blob/master/example/deepseekv3/a16w4-gptq-pergroup/readme.md#%E9%B2%B2%E9%B9%8F%E6%95%B0%E5%AD%A6%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85%E8%AF%B4%E6%98%8E)，进行鲲鹏数学库安装。