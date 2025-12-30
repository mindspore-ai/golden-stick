# Golden Stick PTQ 模型支持列表

本文档列出了 Golden Stick PTQ 框架当前支持的模型。

## 支持的模型生态

Golden Stick 支持两大模型生态：

- **MindOne**：基于 MindSpore 的 Transformers 模型库，支持直接使用 Hugging Face 格式权重。
- **MindFormers**：MindSpore 原生模型库，使用 YAML 配置文件，支持直接使用 Hugging Face 格式权重。

---

## MindOne 生态模型

| 模型系列 | RTN-A16W8 | SmoothQuant | AWQ | GPTQ | OutlierSuppressionLite | A8W4 | A8dynW8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| glm4 | ✅ | ✅ | ✅ | ✅ | | | |
| [glm4v](./glm4v/) | ✅ | ✅ | ✅ | ✅ |✅| |✅|
| qwen3 | ✅ | ✅ | ✅ | ✅ | | | |
| [qwen3_vl](./qwen3vl/) | ✅ | ✅ | | |✅| | |

> A8dynW8即权重8bit perchannel静态量化，激活8bit pertoken动态量化。

---

## MindFormers 生态模型

| 模型系列 | RTN-A16W8 | SmoothQuant | AWQ | GPTQ | OutlierSuppressionLite | A8W4 | A8dynW8 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [deepseek_v3](./deepseekv3/) | ✅ | ✅ | | | ✅ | ✅ | |
| [deepseek_r1](./deepseekv3/) | ✅ | ✅ | | | ✅ | ✅ | |
| qwen3 | ✅ | ✅ | | |✅| | |
| qwen3_moe | ✅ | ✅ | | |✅| | |
| telechat2 | ✅ | ✅ | | | | | |

---

## 参考资源

- [MindOne Transformers](https://gitee.com/mindspore/mindone)
- [MindFormers](https://gitee.com/mindspore/mindformers)
- [Golden Stick 官方文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/index.html)
