# MindSpore Golden Stick

[View English](./README.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [概述](#概述)
    - [设计思路](#设计思路)
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

MindSpore Golden Stick是一个开源的模型压缩算法集，提供了一套统一的算法应用接口，让用户能够统一方便地使用例如量化、剪枝等等模型压缩算法。

MindSpore Golden Stick同时为算法开发者提供修改网络定义的基础设施，在算法和网络定义中间抽象了一层IR，对算法开发者屏蔽具体的网络定义，使其能聚焦与算法逻辑的开发上。

![金箍棒架构图](docs/golden-stick-arch.png)

## 设计思路

1. 提供以用户为中心的API，降低用户学习成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。MindSpore Golden Stick对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些AMC（自动模型压缩技术）、NAS（网络结构搜索）等技术的探索。

2. 提供一些基础设施能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，而很少关注具体的网络定义。MindSpore Golden Stick提供了通过接口修改前端网络定义的能力，让算法开发者聚焦于算法的实现，而不用对不同的网络定义重复造轮子。此外MindSpore Golden Stick还会提供了一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

未来规划

  MindSpore Golden Stick初始版本提供一个稳定的API，并提供一个线性量化算法，一个非线性量化算法和一个结构化剪枝算法。后续会提供更多的算法和更完善的网络支持，调测能力也会在后续版本提供。将来随着算法的丰富，MindSpore Golden Stick还会探索自动模型压缩、硬件感知自动量化和网络结构搜索等能力，敬请期待。

## 安装

MindSpore Golden Stick依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#安装)，再安装MindSpore Golden Stick。可以采用pip安装或者源码编译安装两种方式。

### MindSpore版本依赖关系

由于MindSpore Golden Stick与MindSpore有依赖关系，请按照根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore-Version}/MindSpore/cpu/ubuntu_x86/mindspore-{MindSpore-Version}-cp37-cp37m-linux_x86_64.whl
```

| MindSpore Golden Stick版本 |                             分支                             | MindSpore版本 |
| :---------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

## pip安装

使用pip命令安装，请从[MindSpore Golden Stick下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/golden_stick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Golden Stick安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{ms_version}`表示与MindSpore Golden Stick匹配的MindSpore版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{ms_version}`应写为1.8.0。
> - `{mg_version}`表示MindSpore Golden Stick版本号，例如下载0.1.0版本MindSpore Golden Stick时，`{mg_version}`应写为0.1.0。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/golden-stick)，下载后进入`golden_stick`目录。

```shell
bash build.sh
pip install output/mindspore_gs-0.1.0-py3-none-any.whl
```

其中，`build.sh`为`golden_stick`目录下的编译脚本文件。

## 验证安装是否成功

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
