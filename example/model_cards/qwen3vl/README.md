# Qwen3-VL模型量化

## 1. 介绍

### 1.1  模型简介

[Qwen3-VL](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl) 是由阿里巴巴通义千问团队于2025年9月发布的开源多模态视觉语言模型，也是Qwen系列中性能最强的视觉语言模型。其设计目标不仅是“看到”图像或视频，更能理解内容并驱动行动，实现从“识别”到“推理与执行”的跨越。

### 1.2 量化说明

本指南提供了使用MindSpore Golden Stick对Qwen3-VL模型进行训练后量化（PTQ）的完整流程。通过量化可以显著减少模型存储空间和推理时间，同时尽可能保持模型的精度。

### 1.3 支持的模型

支持的模型：

- **Qwen3-VL-2B-Instruct**：2B参数规模的多模态模型，极致轻量，端侧首选
    - HuggingFace模型路径：[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
    - 特点：响应速度极快，基础视觉问答与描述。在手机等资源受限设备上表现流畅。
- **Qwen3-VL-4B-Instruct**：4B参数规模的多模态模型，轻量高效，平衡之选
    - HuggingFace模型路径：[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
    - 特点：在轻量级模型中保持较好的多模态理解能力，是性价比高的入门选择。
- **Qwen3-VL-8B-Instruct**：8B参数规模的多模态模型，性能与效率的黄金平衡点
    - HuggingFace模型路径：[Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
    - 特点：支持复杂的视觉推理、文档解析和长视频理解，是大多数开发者和研究应用的理想起点。
- **Qwen3-VL-32B-Instruct**：32B参数规模的多模态模型，密集架构的高性能旗舰
    - HuggingFace模型路径：[Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
    - 特点：在多项评测中超越GPT-5mini等竞品，细节感知、复杂推理和跨模态对齐能力接近顶级闭源模型。

### 1.4 支持的量化算法

| 量化算法 | 说明 | 激活量化 | 权重量化 |
|---------|------|---------|---------|
| a8w8_smooth_quant | SmoothQuant算法，通过平滑激活值分布提升量化精度。基础版本的激活/权重8bit量化，吞吐和显存均能提升。| int8 | int8 |
| a8w8_osl | OutlierSuppressionLite算法，为每个矩阵搜索最优超参α值。精度更高的激活/权重8bit量化，吞吐和显存与SmoothQuant一致，但是在量化参数搜索时会耗费更长的时间。| int8 | int8 |

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
git checkout 4be9653
pip install -e .
```

### 2.2 量化权重生成

使用`calibrate.py`脚本进行模型量化。该脚本支持多种量化算法，默认使用`a8w8_osl`算法。

#### 基本用法

下载校准数据：[lmms-lab/textvqa](https://huggingface.co/datasets/lmms-lab/textvqa)

```bash
python calibrate.py -m Qwen/Qwen3-VL-8B-Instruct -q a8w8_osl -d /path/to/textvqa -o ./quant_model_a8w8_osl -b ascend
```

#### 参数说明

- `--model_name` / `-m`: 预训练模型路径或HuggingFace模型名称，默认为`Qwen/Qwen3-VL-8B-Instruct`,也可支持本地模型路径
- `--quant_type` / `-q`: 量化类型，可选值：
    - `a8w8_smooth_quant`: SmoothQuant量化（默认）
    - `a8w8_osl`: OutlierSuppressionLite量化
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
6. **修改config.json**：量化完成后，需要手动在输出目录的 `config.json` 中添加 `"quantization": "golden-stick"` 配置项，参考以下示例：

  ```json
  {
    "model_type": "qwen3_vl",
    ...
    "quantization": "golden-stick"
  }
  ```

#### 注意事项

- **多模态输入**：Qwen3-VL模型需要包含图像和文本的多模态校准数据集
- **量化范围**：默认会排除视觉编码器中的`merger`、`linear_fc2`和语言模型的`lm_head`、`down_proj`。
- **校准样本数**：默认使用200个样本进行校准，可根据需要调整`create_calib_datasets`函数中的`num_samples`参数
- **量化后的模型**：会保存为HuggingFace格式，可直接用于推理

## 3. 精度评估

### 3.1 数据集精度

下表展示了Qwen3-VL-8B-Instruct模型在不同数据集上的精度表现：

| 量化算法 | textvqa | ceval | gsm8k |
|---------|--------------|------------|------|
| bfloat16 (原始模型) | 80.63% | 81.87% | 94.69% |
| a8w8_osl | 80.5% | 82.62% | 94.47% |

**说明**：上述评测结果基于MindSpore Golden Stick r1.4.0版本。实际精度可能因模型版本、数据集版本等因素有所差异。
