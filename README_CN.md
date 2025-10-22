<div align="center">

# MindSpore Golden Stick

#### MindSpore Golden Stick 是MindSpore开源社区的模型压缩工具，支持在昇腾硬件上将Hugging Face的权重进行量化，并在[vllm-MindSpore Plugin](https://gitee.com/mindspore/vllm-mindspore)或[MindSpore Transformers](https://gitee.com/mindspore/mindformers)上进行部署。

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://gitee.com/mindspore/golden-stick)
[![version](https://img.shields.io/badge/release-1.3.0-green)](https://gitee.com/mindspore/golden-stick/releases)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)

[**架构**](docs/zh_cn/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**流程**](docs/zh_cn/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**文档**](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/index.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**问题反馈**](https://gitee.com/mindspore/golden-stick/issues)

[English](README.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[中文](README_CN.md)

<div align="left">

MindSpore Golden Stick（后面简称金箍棒）是MindSpore团队和华为诺亚团队联合设计开发的一个模型压缩工具。我们有两大目标：一方面我们希望构建MindSpore开源生态的模型压缩能力，并提供简洁易用的接口方便用户提升MindSpore网络的部署效率；另一方面我们希望屏蔽框架和硬件的复杂性，为模型压缩算法提供扩展性良好的基础能力。

- **SoTA算法：** 金箍棒中的模型压缩算法主要有两大来源，一个是业界的SoTA的算法，我们会持续在MindSpore生态上跟进；另一个是华为算法团队提供的创新性算法；
- **接口易用：** 金箍棒提供类Transformers的接口，并且支持直接将Hugging Face社区的权重进行压缩，输出的权重同样符合Hugging Face社区权重格式；
- **分层解耦：** 金箍棒致力于构建易用的算法预研平台，我们对框架做了分层和模块化设计，一方面屏蔽框架和硬件的复杂性，另一方面方便算法工程师在算法不同层级快速进行创新和实验。

## 最新消息🔥

* [2025/09] OutlierSuppressionLite提供更高精度的A8W8量化能力。
* [2025/09] 结合OutlierSuppressionLite和GPTQ算法，实现DeepSeekV3/R1网络的A8W4量化，满血版DeepSeek部署门槛进一步降低。量化权重详见[魔乐社区权重](https://modelers.cn/models/MindSpore-Lab/R1-0528-A8W4)。
* [2025/09] 支持[Transformers-Like-API](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html#mindspore_gs.ptq.AutoQuantForCausalLM)，并支持保存出huggingface格式的权重，详见 [BaseQuantForCausalLM](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/ptq/mindspore_gs.ptq.BaseQuantForCausalLM.html#mindspore_gs.ptq.BaseQuantForCausalLM.save_quantized) 接口。
* [2025/06] 支持对DeepSeekV3/R1网络进行SmoothQuant-8bit、GPTQ-4bit量化。

## 安装

请参考[安装教程](docs/zh_cn/install.md)。

## 快速入门

以一个简单的算法[Simulated Quantization (SimQAT)](mindspore_gs/quantization/simulated_quantization/README_CN.md) 作为例子，演示如何在训练中应用金箍棒中的算法。

## 文档

<table text-align="center" width="100%">
  <thead>
  <tr>
    <th colspan="60"><div align="center">概览</div></th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="20" align="center"><a href="docs/zh_cn/design.md">架构</a></td>
      <td colspan="20" align="center"><a href="docs/zh_cn/design.md">流程</a></td>
      <td colspan="20" align="center"><a href="example/">样例</a></td>
    </tr>
    <tr>
      <td colspan="30" align="center"><a href="https://www.mindspore.cn/golden_stick/docs/zh-CN/master/ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html#mindspore_gs.ptq.AutoQuantForCausalLM">Transformers like APIs🔥</a></td>
      <td colspan="30" align="center"><a href="https://www.mindspore.cn/golden_stick/docs/zh-CN/master/mindspore_gs.ptq.html">APIs</a></td>
    </tr>
  <thead>
    <tr>
      <th colspan="60"><a href="mindspore_gs/ptq/README_CN.md"><div align="center">训练后量化</div></a></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">RoundToNearest-A16W8</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">SmoothQuant-A8W8</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">AWQ-A16W4</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">GPTQ-A16W4</a></td>
    <tr>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">QoQ-A8W4🔥</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">FAQuant(demo)</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">Dynamic Quantization</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">KVCacheInt8</a></td>
    </tr>
    <tr>
      <td colspan="30" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">OutlierSuppressionLite🔥</a></td>
      <td colspan="30" align="center"><a href="mindspore_gs/ptq/ptq/README_CN.md">OutlierSuppressionPlus(demo)</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="60"><div align="center">其他</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="20" align="center">自动策略搜优</td>
      <td colspan="20" align="center">伪量化评测</td>
      <td colspan="20" align="center">昇腾硬件适配层</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="60"><div align="center">End Of Life</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="30" align="center"><a href="mindspore_gs/quantization/simulated_quantization/README_CN.md">QAT-SimQAT</a></td>
      <td colspan="30" align="center"><a href="mindspore_gs/quantization/slb/README_CN.md">QAT-SLB</a></td>
    <tr>
    </tr>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/scop/README_CN.md">pruner-SCOP</a></td>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/uni_pruning/README.md">pruner-uni_pruning(demo)</a></td>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/heads/lrp/README.md">pruner-LRP(demo)</a></td>
    <tr>
    </tr>
      <td colspan="60" align="center"><a href="mindspore_gs/ghost/README_CN.md">Ghost</a></td>
    </tr>
  </tbody>
</table>

### 模型部署

金箍棒模型压缩得到的是Hugging Face格式的权重，推荐在[vllm-MindSpore Plugin](https://gitee.com/mindspore/vllm-mindspore)或者[MindSpore Transformers](https://gitee.com/mindspore/mindformers)上进行部署，也可以尝试在Pytorch、ONNX Runtime、TensorRT等主流框架上进行部署。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

🎯视频会议：https://meeting.tencent.com/dm/U5EJCKl1FP8z

📬SIG：https://www.mindspore.cn/sig/LLM%20Inference%20Serving

📍微信群：https://gitee.com/mindspore/golden-stick/issues/ID2UGQ

## 贡献

请参考 [CONTRIBUTING](./CONTRIBUTING.md) 文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作，请通过 [Issue](https://gitee.com/mindspore/golden-stick/issues) 来告知我们您遇到的任何Bug，或提交您的特性需求、改进建议、技术方案。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)
