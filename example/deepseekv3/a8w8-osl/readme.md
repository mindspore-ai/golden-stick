# DeepSeekV3(R1)网络OutlierSuppressionLite(OSL) A8W8量化算法指南

本指南基于单机16卡，如果使用双机16卡，请将msrun命令替换为双机16卡形式。本指南将使用ceval数据集校准。

## 1. 运行前准备

请提前下载bf16格式的deepseek v3/r1权重一套、ceval数据集一套。
运行前请确保备有至少1.3T空闲磁盘空间，其中约630G将用于存储最终A8W8量化后权重，另约650G将用于量化过程中分布式权重存储，其余空间用于日志记录。

## 2. 修改配置文件

一键量化脚本默认使用的yaml配置文件为当前目录下的calibrate_deepseek3_671b.yaml，需要将yaml文件的load_checkpoint、pretrained_model_dir参数修改为bf16权重路径。

## 3. 进行量化校准

执行下述命令，启动一键式量化校准：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate.sh /path/to/ceval-dataset/dev
```

## 4. 获取量化校准结果

执行量化校准后，路径中的文件如下表。

| 路径 | 文件说明 |
| --- | --- |
| quantized_model/ | 分布式权重输出路径 |
| quantized_model/rank_*/ | 分布式权重，权重合一后即废弃 |
| quantized_model/quantization_description.json | 量化策略描述 |
| quantized_model_unified/ | 合一权重输出路径 |
| quantized_model_unified/*.safetensors | 合一权重，可用于推理 |
| quantized_model_unified/*.safetensors.index.json | 合一权重映射关系文件 |
| unify.log | 权重合一日志 |
| calibrate.log | 校准日志 |
| log_calibrate/ | 各npu校准日志 |

执行下述步骤，制作可用于推理的权重：

1. 获取量化后权重文件：将quantized_model_unified/*.safetensors放置到目标路径。

2. 获取量化后权重的描述文件：quantized_model/quantization_description.json和quantized_model_unified/*.safetensors.index.json放置到目标路径。

3. 制作权重config：将原始浮点权重中的config.json拷贝至目标路径，并将文件中的"quantization_config"配置修改为：

   ```json
   "quantization_config": {
     "quant_method": "golden-stick"
   }
   ```

4. 复用浮点tokenizer：将原始浮点权重中的tokenizer_config.json和tokenizer.json拷贝至目标路径。
