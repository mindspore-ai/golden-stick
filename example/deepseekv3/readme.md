# DeepSeekR1网络量化算法指南

本指南基于单机16卡，如果使用双机16卡，请将msrun命令替换为双机16卡形式。

运行前请检查yaml配置中的tp并行数，load_checkpoint配置，tokenizer配置是否合理。

## 0. 首先使用动态图测试浮点网络对话精度

为了保证环境上安装包和下载的权重的正确性，首先使用动态图将DeepSeekR1浮点网络在环境上进行一次对话，查看结果是否正常。命令如下：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash pynative_infer_single.sh /path/to/mindformers "" worker_num /path/to/yaml
```

也可以缺省卡数为16，yaml配置文件为mindformers目录下默认的config：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash pynative_infer_single.sh /path/to/mindformers
```

## 1. 进行算法校准阶段

使用数据集进行量化校准。命令如下：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant worker_num /path/to/yaml
```

也可以缺省卡数为16，yaml配置文件为mindformers目录下默认的config：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash calibrate_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant
```

## 2. 使用动态图测试量化网络对话精度

首先测试在动态图下，量化后的网络是否对话ok，命令如下：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash pynative_infer_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant worker_num /path/to/yaml
```

也可以缺省卡数为16，yaml配置文件为mindformers目录下默认的config：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash pynative_infer_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant
```

## 3. 测试图模式下的量化网络对话精度

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant worker_num /path/to/yaml
```

也可以缺省卡数为16，yaml配置文件为mindformers目录下默认的config：

```bash
ASCEND_RT_VISIBLE_DEVICES=xxx bash quant_infer_single.sh /path/to/mindformers a16w8/a8dynw8/smoothquant/awq-a16w4/awq-a16w8/dsquant
```
