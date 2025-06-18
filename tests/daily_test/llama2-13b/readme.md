# Llama网络daily-test脚本使用指南

`daily_test-llama.sh`提供了在Llama网络上使用PTQ相关算法进行量化与评测的能力。可在8卡机器上并行执行量化与评测命令，提高测试效率。

## 启动命令

启动时依次传入下列参数：

- `/path/to/bf16_model_path`: 浮点模型权重路径

- `/path/to/vocab_file`: vocab_file所在路径

- `ms_pkg_link`: mindspore版本包路径(非必填)。

- `mf_pkg_link`: mindformer版本包路径(非必填)。

运行命令如下：

(1) 使用默认的mindspore和mindformers版本包。

```bash
bash daily_test_llama.sh "/path/to/bf16_model_path" "/path/to/a8w8_model_path"
```

(2) 使用给定的mindspore和mindformers版本包。

```bash
bash daily_test_llama.sh "/path/to/bf16_model_path" "/path/to/a8w8_model_path" "ms_pkg_link" "mf_pkg_link"
```