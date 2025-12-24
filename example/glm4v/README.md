# GLM-4.1v模型量化

## 1. 介绍

### 1.1 GLM-4.1V 模型简介

GLM-4.1V（GLM-4.1 Vision）是由智谱AI开发的多模态大语言模型，是GLM-4系列中的视觉理解版本。该模型能够同时处理图像和文本输入，实现视觉-语言的多模态理解和生成任务。

#### 模型特点

- **多模态能力**：支持图像和文本的联合输入，能够理解图像内容并基于图像进行对话
- **视觉编码器**：采用专门的视觉编码器（Vision Encoder）处理图像输入，提取视觉特征
- **语言模型**：基于GLM-4架构的语言模型，具备强大的文本理解和生成能力
- **统一架构**：通过视觉-语言对齐机制，实现图像和文本的统一表示和处理

### 1.2 量化说明

本指南提供了使用MindSpore Golden Stick对GLM-4.1V模型进行训练后量化（PTQ）的完整流程。通过量化可以显著减少模型存储空间和推理时间，同时尽可能保持模型的精度。

### 1.3 支持的模型

当前已验证的模型：

- **GLM-4.1V-9B-Thinking**：9B参数规模的多模态模型，支持图像和文本的联合理解与生成
    - HuggingFace模型路径：[zai-org/GLM-4.1V-9B-Thinking](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking)
    - 特点：支持思维链推理，能够进行复杂的视觉-语言理解任务

### 1.4 支持的量化算法

| 量化算法 | 说明 | 激活量化 | 权重量化 |
|---------|------|---------|---------|
| a8w8_smooth_quant | SmoothQuant算法，通过平滑激活值分布提升量化精度。基础版本的激活/权重8bit量化，吞吐和显存均能提升。 | int8 | int8 |
| a8w8_osl | OutlierSuppressionLite算法，为每个矩阵搜索最优超参α值。精度更高的激活/权重8bit量化，吞吐和显存与SmoothQuant一致，但是在量化参数搜索时会耗费更长的时间。 | int8 | int8 |
| a16w8 | 仅权重量化，激活保持FP16。显存有收益，吞吐没有收益。 | FP16 | int8 |
| a8dyw8 | 动态per-token激活量化。激活pertoken量化，精度更高。 | int8 (per-token) | int8 |
| a16w4_awq | AWQ算法，激活感知权重量化。权重4比特量化，显存占用更少。 | FP16 | qint4x2 (per-group) |
| a16w4_gptq | GPTQ算法，逐层补偿量化误差。权重4比特量化，显存占用更少。 | FP16 | qint4x2 (per-group) |

## 2. 快速开始

### 2.1 环境准备

确保如下依赖已安装：

| mindspore | ascend driver | firmware | cann toolkit/kernel |
|--------------|--------------|----------|-------------------|
| [2.7.1](https://www.mindspore.cn/install) | 24.1.RC3.b080 | 7.5.T11.0.B088 | [8.0.RC3.beta1](https://www.hiascend.com/developer/download/community/result?module=cann) |

(1) MindSpore Golden stick安装:

```bash
git clone https://gitee.com/mindspore/golden-stick.git
cd golden-stick
pip install -e .
```

(2) mindone安装:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

### 2.2 量化权重生成

使用`quant.py`脚本进行模型量化。该脚本支持多种量化算法，默认使用`a8w8_osl`算法。

#### 基本用法

下载校准数据：[lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa)

```bash
python quant.py -m zai-org/GLM-4.1V-9B-Thinking -q a8w8_osl -d /path/to/textvqa -o ./quant_model_a8w8_osl -b ascend
```

#### 参数说明

- `--model_name` / `-m`: 预训练模型路径或HuggingFace模型名称，默认为`zai-org/GLM-4.1V-9B-Thinking`,也可支持本地模型路径
- `--quant_type` / `-q`: 量化类型，可选值：
    - `a8w8_smooth_quant`: SmoothQuant量化（默认）
    - `a8w8_osl`: OutlierSuppressionLite量化
    - `a16w8`: 仅权重量化
    - `a8dyw8`: 动态per-token激活量化
    - `a16w4_awq`: AWQ量化
    - `a16w4_gptq`: GPTQ量化
- `--calib_dataset_path` / `-d`: 量化校准数据集路径（支持包含图像和文本的多模态数据集）
- `--output_path` / `-o`: 量化模型保存路径，默认为`./quant_model`
- `--backend` / `-b`: 后端目标，可选值：
    - `ascend`: 昇腾后端（默认）
    - `none`: 通用后端

#### 量化流程

1. **创建校准数据集**：脚本会从指定的数据集路径加载多模态样本（包含图像和文本）用于校准，默认使用200个样本
2. **创建PTQ配置**：根据指定的量化类型创建相应的PTQ配置，自动排除视觉编码器和语言模型头部
3. **加载模型**：使用`AutoQuantForCausalLM.from_pretrained()`加载预训练的多模态模型
4. **校准**：使用校准数据集对模型进行量化校准
5. **保存量化模型**：将量化后的模型保存到指定路径，格式为HuggingFace格式，支持指定后端类型

#### 注意事项

- **多模态输入**：GLM-4.1V模型需要包含图像和文本的多模态校准数据集
- **量化范围**：默认会排除视觉编码器（`visual`）和语言模型头部（`lm_head`），仅对语言模型主体进行量化
- **校准样本数**：默认使用200个样本进行校准，可根据需要调整`create_calib_datasets`函数中的`num_samples`参数
- **量化后的模型**：会保存为HuggingFace格式，可直接用于推理
- **校准时间**：部分量化算法（如AWQ、GPTQ）需要较长的校准时间，建议使用较少的校准样本进行快速测试

## 3. 精度评估

### 3.1 数据集精度

下表展示了GLM-4.1V-9B-Thinking模型在不同数据集上的精度表现：

| 量化算法 | textvqa | ceval | gsm8k |
|---------|--------------|------------|------|
| bfloat16 (原始模型) | 76.3% | 79.12% | 92.72% |
| a8w8_osl | 75.96% | 78.31% | 91.96% |

**说明**：上述评测结果基于MindSpore Golden Stick最新版本。实际精度可能因模型版本、数据集版本等因素有所差异。

## 4. 常见问题

### Q1: 量化过程中内存不足怎么办？

A: 可以尝试以下方法：

- 减少校准数据集的样本数量（修改`create_calib_datasets`函数中的`num_samples`参数）
- 使用更小的模型进行测试
- 确保有足够的系统内存

### Q2: 量化后的模型精度下降明显怎么办？

A: 可以尝试：

- 使用`a8w8_osl`算法，通常精度更高
- 增加校准数据集的样本数量

### Q3: 如何自定义校准数据集？

A: 修改`create_calib_datasets`函数，可以：

- 更换数据集路径（支持HuggingFace数据集或本地JSONL文件）
- 修改数据预处理逻辑（`preprocess_and_tokenizer`函数）
- 调整样本数量（修改`num_samples`参数，默认200个样本）

## 5. 参考资源

- [MindSpore Golden Stick PTQ文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/ptq/ptq.html)
- [GLM-4技术报告](https://github.com/THUDM/GLM-4)
- [MindOne Transformers文档](https://github.com/mindspore-lab/mindone)
