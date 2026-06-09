<div align="center">

# MindSpore Golden Stick

**MindSpore Golden Stick is a model compression tool for the MindSpore open source community, supporting quantization of Hugging Face weights on Ascend hardware and deployment on [vLLM-MindSpore Plugin](https://atomgit.com/mindspore/vllm-mindspore) or [MindSpore Transformers](https://atomgit.com/mindspore/mindformers).**

[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://atomgit.com/mindspore/golden-stick)
[![version](https://img.shields.io/badge/release-1.3.0-green)](https://atomgit.com/mindspore/golden-stick/releases)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://atomgit.com/mindspore/golden-stick/blob/master/LICENSE)

[**Architecture**](docs/en/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Workflow**](docs/en/design.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Documentation**](https://www.mindspore.cn/golden_stick/docs/en/master/index.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[**Issue Feedback**](https://atomgit.com/mindspore/golden-stick/issues)

[English](README.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[中文](README_CN.md)

<div align="left">

MindSpore Golden Stick is a model compression tool jointly designed and developed by MindSpore team and Huawei Noah's Ark Lab. We have two main goals: first, to be a model compression tool that provides concise interfaces and rich algorithm libraries to improve the deployment efficiency of MindSpore networks; second, to be an algorithm research platform with flexible configuration interfaces, modular algorithm libraries, and a framework that supports rapid customization, facilitating algorithm researchers to quickly practice innovation. Specifically:

- **Multi-level APIs:** Provides different level APIs, offering both ease of use and flexibility, lowering the barrier to entry while retaining algorithm customization capabilities;
- **Rich and Modular Algorithm Library:** Provides rich SoTA compression algorithms and supports flexible modular combinations;
- **Highly Extensible Framework Architecture:** Layered decoupling that shields the complexity of hardware and frameworks, while supporting integration of custom algorithm components to build customized compression pipelines with flexible APIs.

## What's New🔥

* [2025/12] **v1.4.0 Release**: Completed framework pluginization refactoring with MindONE backend support, integrating mainstream quantization algorithms including OSL, SmoothQuant, AWQ, GPTQ, A16W8, A8dynW8, and A8W4, validated on models such as glm4v and qwen3.
* [2025/12] **Multimodal Model Quantization**: Added support for quantization of multimodal understanding models, successfully validating the OSL-A8W8 quantization scheme for the qwen3-vl network under MindONE framework.
* [2025/09] OutlierSuppressionLite provides higher precision A8W8 quantization capabilities.
* [2025/09] Combined OutlierSuppressionLite and GPTQ algorithms to achieve A8W4 quantization for DeepSeekV3/R1 networks, further lowering the deployment threshold for full-featured DeepSeek. Quantized weights can be found at [Modelers](https://modelers.cn/models/MindSpore-Lab/R1-0528-A8W4).
* [2025/09] Support for [Transformers-Like-API](https://www.mindspore.cn/golden_stick/docs/en/master/ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html#mindspore_gs.ptq.AutoQuantForCausalLM) and support for saving weights in Hugging Face format, see [BaseQuantForCausalLM](https://www.mindspore.cn/golden_stick/docs/en/master/ptq/mindspore_gs.ptq.BaseQuantForCausalLM.html#mindspore_gs.ptq.BaseQuantForCausalLM.save_quantized) interface for details.
* [2025/06] Support for SmoothQuant-8bit and GPTQ-4bit quantization of DeepSeekV3/R1 networks.

## Installation

Please refer to [Installation Tutorials](docs/en/install.md).

## Quick Start

Take [Simulated Quantization (SimQAT)](mindspore_gs/quantization/simulated_quantization/README.md) as an example for demonstrating how to use MindSpore Golden Stick.

## Documentation

<table text-align="center" width="100%">
  <thead>
  <tr>
    <th colspan="60"><div align="center">Overview</div></th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="20" align="center"><div>Architecture</div></td>
      <td colspan="20" align="center"><div>Workflow</div></td>
      <td colspan="20" align="center"><a href="example/">Examples</a></td>
    </tr>
    <tr>
      <td colspan="30" align="center"><a href="https://www.mindspore.cn/golden_stick/docs/en/master/ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html#mindspore_gs.ptq.AutoQuantForCausalLM">Transformers like APIs🔥</a></td>
      <td colspan="30" align="center"><a href="https://www.mindspore.cn/golden_stick/docs/en/master/mindspore_gs.ptq.html">APIs</a></td>
    </tr>
  <thead>
    <tr>
      <th colspan="60"><a href="mindspore_gs/ptq/README.md"><div align="center">Post-Training Quantization</div></a></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">RoundToNearest-A16W8</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">SmoothQuant-A8W8</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">AWQ-A16W4</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">GPTQ-A16W4</a></td>
    <tr>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">QoQ-A8W4🔥</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">FAQuant(demo)</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">Dynamic Quantization</a></td>
      <td colspan="15" align="center"><a href="mindspore_gs/ptq/ptq/README.md">KVCacheInt8(demo)</a></td>
    </tr>
    <tr>
      <td colspan="30" align="center"><a href="mindspore_gs/ptq/ptq/README.md">OutlierSuppressionLite🔥</a></td>
      <td colspan="30" align="center"><a href="mindspore_gs/ptq/ptq/README.md">OutlierSuppressionPlus(demo)</a></td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="60"><div align="center">Others</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="20" align="center">Auto Quantization Strategy</td>
      <td colspan="20" align="center">Fake Quant Evaluation</td>
      <td colspan="20" align="center">Ascend Hardware Adapter layer</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="60"><div align="center">End Of Life</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="30" align="center"><a href="mindspore_gs/quantization/simulated_quantization/README.md">QAT-SimQAT</a></td>
      <td colspan="30" align="center"><a href="mindspore_gs/quantization/slb/README.md">QAT-SLB</a></td>
    <tr>
    </tr>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/scop/README.md">pruner-SCOP</a></td>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/uni_pruning/README.md">pruner-uni_pruning(demo)</a></td>
      <td colspan="20" align="center"><a href="mindspore_gs/pruner/heads/lrp/README.md">pruner-LRP(demo)</a></td>
    <tr>
    </tr>
      <td colspan="60" align="center"><a href="mindspore_gs/ghost/README.md">Ghost</a></td>
    </tr>
  </tbody>
</table>

### Model Deployment

The model compression results from Golden Stick are weights in Hugging Face format. It is recommended to deploy them on [vLLM-MindSpore Plugin](https://atomgit.com/mindspore/vllm-mindspore) or [MindSpore Transformers](https://atomgit.com/mindspore/mindformers). You can also try deploying them on mainstream frameworks such as PyTorch, ONNX Runtime, TensorRT, etc.

## Community

### Governance

[MindSpore Open Governance](https://atomgit.com/mindspore/community/blob/master/governance.md)

### Communication

🎯Video Conference：https://meeting.tencent.com/dm/U5EJCKl1FP8z

📬SIG：https://www.mindspore.cn/sig/LLM%20Inference%20Serving

📍WeChat Group：https://atomgit.com/mindspore/golden-stick/issues/ID2UGQ

## Contributing

Please read [CONTRIBUTING](./CONTRIBUTING.md) for details on setting up development environments, testing functions, and submitting PR.

We welcome and value any form of contribution and cooperation. Please use [Issue](https://atomgit.com/mindspore/golden-stick/issues) to inform us of any bugs you encounter, or to submit your feature requests, improvement suggestions, and technical solutions.

## License

[Apache License 2.0](https://atomgit.com/mindspore/golden-stick/blob/master/LICENSE)
