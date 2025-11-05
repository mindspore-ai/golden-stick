<div align="center">

# MindSpore Golden Stick

**MindSpore Golden Stick 是MindSpore开源社区的模型压缩工具，支持在昇腾硬件上将Hugging Face的权重进行量化，并在[vLLM-MindSpore Plugin](https://gitee.com/mindspore/vllm-mindspore)或[MindSpore Transformers](https://gitee.com/mindspore/mindformers)上进行部署。**

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://gitee.com/mindspore/golden-stick)
[![version](https://img.shields.io/badge/release-1.3.0-green)](https://gitee.com/mindspore/golden-stick/releases)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)

[**架构**](docs/zh_cn/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**流程**](docs/zh_cn/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**文档**](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/index.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**问题反馈**](https://gitee.com/mindspore/golden-stick/issues)

[English](README.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[中文](README_CN.md)

<div align="left">

MindSpore Golden Stick（后面简称金箍棒）是MindSpore团队和华为诺亚团队联合设计开发的一个模型压缩工具。我们有两大目标：一是做一个模型压缩工具，提供简洁的接口以及丰富的算法库，以提升MindSpore网络的部署效率；二是做一个算法研究平台，提供灵活的配置接口和积木化的算法库，并支持快速自定义的框架，方便算法研究员快速实践创新。具体来说：

- **多层级API**：提供不同level的API，兼顾易用性和灵活性，降低使用门槛，同时保留算法定制化的能力；
- **丰富且模块化的算法库**：提供丰富的SoTA压缩算法，并且支持灵活模块化组合；
- **高度可扩展的框架架构**：分层解耦，屏蔽硬件和框架的复杂性，同时支持集成自定义算法组件，配合灵活的API构建定制化压缩流水线。

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
      <th colspan="60"><div align="center">生命周期终止</div></th>
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

金箍棒模型压缩得到的是Hugging Face格式的权重，推荐在[vLLM-MindSpore Plugin](https://gitee.com/mindspore/vllm-mindspore)或者[MindSpore Transformers](https://gitee.com/mindspore/mindformers)上进行部署，也可以尝试在Pytorch、ONNX Runtime、TensorRT等主流框架上进行部署。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

🎯视频会议：https://meeting.tencent.com/dm/U5EJCKl1FP8z

📬SIG：https://www.mindspore.cn/sig/LLM%20Inference%20Serving

📍微信群：https://gitee.com/mindspore/golden-stick/issues/ID2UGQ

## 贡献

请参考 [CONTRIBUTING](./CONTRIBUTING_CN.md) 文档了解更多关于开发环境搭建、功能测试以及 PR 提交规范的信息。

我们欢迎并重视任何形式的贡献与合作，请通过 [Issue](https://gitee.com/mindspore/golden-stick/issues) 来告知我们您遇到的任何Bug，或提交您的特性需求、改进建议、技术方案。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)
