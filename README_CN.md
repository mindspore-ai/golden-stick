# MindSpore Golden Stick

[View English](./README.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [概述](#概述)
    - [设计思路](#设计思路)
    - [未来规划](#未来规划)
    - [安装](#安装)
        - [MindSpore版本依赖关系](#mindSpore版本依赖关系)
        - [pip安装](#pip安装)
        - [源码编译安装](#源码编译安装)
        - [验证](#验证安装是否成功)
    - [快速入门](#快速入门)
    - [文档](#文档)
        - [开发者教程](#开发者教程)
    - [社区](#社区)
        - [治理](#治理)
        - [交流](#交流)
    - [贡献](#贡献)
    - [许可证](#许可证)

<!-- /TOC -->

## 概述

MindSpore Golden Stick是华为诺亚团队和华为MindSpore团队联合设计开发的一个模型压缩算法集。MindSpore Golden Stick的架构图如下图所示，分为五个部分：

1. 底层的MindSpore Rewrite模块提供修改前端网络的能力，基于此模块提供的接口，算法开发者可以按照特定的规则对MindSpore的前端网络做节点和拓扑关系的增删查改；

2. 基于MindSpore Rewite这个基础能力，MindSpore Golden Stick会提供各种类型的算法，比如SimQAT算法、SLB量化算法、SCOP剪枝算法等；

3. 在算法的更上层，MindSpore Golden Stick还规划了如AMC（自动模型压缩技术）、NAS（网络结构搜索）、HAQ（硬件感知的自动量化）等高阶技术；

4. 为了方便开发者分析调试算法，MindSpore Golden Stick提供了一些工具，如Visualization工具（可视化工具）、Profiler工具（逐层分析工具）、Summary工具（算法压缩效果分析工具）等；

5. 在最外层，MindSpore Golden Stick封装了一套简洁的用户接口。

![金箍棒架构图](docs/images/golden-stick-arch.png)

> 架构图是MindSpore Golden Stick的全貌，其中包含了当前已经实现的功能以及规划在RoadMap中能力。具体开放的功能可以参考对应版本的ReleaseNotes。

## 设计思路

MindSpore Golden Stick除了提供丰富的模型压缩算法外，一个重要的设计理念是针对业界种类繁多的模型压缩算法，提供给用户一个尽可能统一且简洁的体验，降低用户的算法应用成本。MindSpore Golden Stick通过两个举措来实现该理念：

1. 统一的算法接口设计，降低用户应用成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。MindSpore Golden Stick对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些AMC、NAS、HAQ等高阶技术的探索。

2. 提供前端网络修改能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，如感知量化算法往往在网络中的Conv2d、Conv2d + BatchNorm2d或者Conv2d + BatchNorm2d + Relu结构上插入伪量化节点。MindSpore Golden Stick提供了通过接口修改前端网络的能力，算法开发者可以基于此能力制定通用的改图规则去实现算法逻辑，而不需要对每个特定的网络都实现一遍算法逻辑算法。此外MindSpore Golden Stick还会提供了一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

## 未来规划

  MindSpore Golden Stick初始版本提供一套稳定的API，并提供一个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持，调测能力也会在后续版本提供。将来随着算法的丰富，MindSpore Golden Stick还会探索自动模型压缩、硬件感知自动量化和网络结构搜索等能力，敬请期待。

## 安装

### 环境限制

下表列出了安装、编译和运行MindSpore Golden Stick所需的系统环境：

| 软件名称 |  版本   |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.7-3.9 |

> 其他的三方依赖请参考[requirements文件](https://gitee.com/mindspore/golden-stick/blob/r0.1/requirements.txt)。
> 当前MindSpore Golden Stick仅能在Ubuntu18.04上运行。

### MindSpore版本依赖关系

MindSpore Golden Stick依赖MindSpore训练推理框架，请按照根据下表中所指示的对应关系，并参考[MindSpore安装指导](https://mindspore.cn/install)安装对应版本的MindSpore：

| MindSpore Golden Stick版本 |                             分支                             | MindSpore版本 |
| :---------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

安装完MindSpore后，继续安装MindSpore Golden Stick。可以采用pip安装或者源码编译安装两种方式。

### pip安装

使用pip命令安装，请从[MindSpore Golden Stick下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/golden_stick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Golden Stick安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{ms_version}`表示与MindSpore Golden Stick匹配的MindSpore版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{ms_version}`应写为1.8.0。
> - `{mg_version}`表示MindSpore Golden Stick版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{mg_version}`应写为0.1.0。

### 源码编译安装

下载[源码](https://gitee.com/mindspore/golden-stick)，下载后进入`golden_stick`目录。

```shell
bash build.sh
pip install output/mindspore_gs-0.1.0-py3-none-any.whl
```

其中，`build.sh`为`golden_stick`目录下的编译脚本文件。

### 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_gs
```

## 快速入门

以一个简单的算法[Simulated Quantization (SimQAT)](https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_zh_cn/quantization/simqat.md) 作为例子，演示如何在训练中应用金箍棒中的算法。

## 文档

### 开发者教程

有关安装指南、教程和API的更多详细信息，请参阅[用户文档]。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。

## 贡献

欢迎参与贡献。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)
