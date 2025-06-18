# DeepSeekR1网络daily-test脚本使用指南

`daily_test-ds.sh`提供了A8W8量化的评测，以及基于gptq算法的A16W4量化和评测能力。

## 启动命令

启动时依次传入下列参数：

- `/path/to/bf16_model_path`: 浮点模型权重路径

- `/path/to/a8w8_model_path`: A8W8模型权重路径

- `/path/to/vocab_file`: vocab_file所在路径

- `/path/to/tokenizer_file`: totokenizer_file所在路径。

脚本会依次执行DeepSeekR1的A8W8量化模型推理，基于gptq算法的A16W6模型量化和推理。

```bash
bash daily_test_ds.sh "/path/to/bf16_model_path" "/path/to/a8w8_model_path" "/path/to/vocab_file" "/path/to/tokenizer_file"
```