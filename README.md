# MindSpore Golden Stick

[查看中文](./README_CN.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [Overview](#overview)
    - [Design Features](#design-features)
    - [Future Roadmap](#future-roadmap)
    - [Installation](#installation)
        - [Version dependency](#version-dependency)
        - [Installing from pip command](#installing-from-pip-command)
        - [Installing from source code](#installing-from-source-code)
        - [Verification](#verification)
    - [Quick Start](#quick-start)
    - [Documents](#documents)
        - [Developer Guide](#developer-guide)
    - [Community](#community)
        - [Governance](#governance)
        - [Communication](#communication)
    - [Contributions](#contributions)
    - [License](#license)

<!-- /TOC -->

## Overview

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. The architecture diagram of MindSpore Golden Stick is shown in the figure below, which is divided into five parts:

1. The underlying MindSpore Rewrite module provides the ability to modify the front-end network. Based on the interface provided by this module, algorithm developers can add, delete, query and modify the nodes and topology relationships of the MindSpore front-end network according to specific rules;

2. Based on MindSpore Rewrite, MindSpore Golden Stick will provide various types of algorithms, such as SimQAT algorithm, SLB quantization algorithm, SCOP pruning algorithm, etc.;

3. At the upper level of the algorithm, MindSpore Golden Stick also plans advanced technologies such as AMC (AutoML for Model Compression), NAS (Neural Architecture Search), and HAQ (Hardware-aware Automated Quantization). This feature will be provided in future;

4. In order to facilitate developers to analyze and debug algorithms, MindSpore Golden Stick provides some tools, such as visualization tool, profiler tool, summary tool, etc. This feature will be provided in future;

5. In the outermost layer, MindSpore Golden Stick encapsulates a set of concise user interface.

![MindSpore_GS_Architecture](docs/images/golden-stick-arch.png)

> The architecture diagram is the overall picture of MindSpore Golden Stick, which includes the features that have been implemented in the current version and the capabilities planned in RoadMap. Please refer to release notes for available features in current version.

## Design Features

In addition to providing rich model compression algorithms, an important design concept of MindSpore Golden Stick is try to provide users with the most unified and concise experience for a wide variety of model compression algorithms in the industry, and reduce the cost of algorithm application for users. MindSpore Golden Stick implements this philosophy through two initiatives:

1. Unified algorithm interface design to reduce user application costs:

   There are many types of model compression algorithms, such as quantization-aware training algorithms, pruning algorithms, matrix decomposition algorithms, knowledge distillation algorithms, etc. In each type of compression algorithm, there are also various specific algorithms, such as LSQ and PACT, which are both quantization-aware training algorithms. Different algorithms are often applied in different ways, which increases the learning cost for users to apply algorithms. MindSpore Golden Stick sorts out and abstracts the algorithm application process, and provides a set of unified algorithm application interfaces to minimize the learning cost of algorithm application. At the same time, this also facilitates the exploration of advanced technologies such as AMC, NAS and HAQ based on the algorithm ecology.

2. Provide front-end network modification capabilities to reduce algorithm development costs：

   Model compression algorithms are often designed or optimized for specific network structures. For example, perceptual quantization algorithms often insert fake-quantization nodes on the Conv2d, Conv2d + BatchNorm2d, or Conv2d + BatchNorm2d + Relu structures in the network. MindSpore Golden Stick provides the ability to modify the front-end network through API. Based on this ability, algorithm developers can formulate general network transform rules to implement the algorithm logic without needing to implement the algorithm logic for each specific network. In addition, MindSpore Golden Stick also provides some debugging capabilities, including visualization tool, profiler tool, summary tool, aiming to help algorithm developers improve development and research efficiency, and help users find algorithms that meet their needs.

### Future Roadmap

The current release version of MindSpore Golden Stick provides a stable API and provides a linear quantization algorithm, a nonlinear quantization algorithm and a structured pruning algorithm. More algorithms and better network support will be provided in the future version, and debugging capabilities will also be provided in subsequent versions. With the enrichment of algorithms in the future, MindSpore Golden Stick will also explore capabilities such as AMC, HAQ, NAS, etc., so stay tuned.

## Installation

### Environmental restrictions

The following table lists the environment required for installing, compiling and running MindSpore Golden Stick:

| software | version  |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.7-3.9 |

> Please refer to [requirements](https://gitee.com/mindspore/golden-stick/blob/r0.1/requirements.txt) for other third party dependencies.
> MindSpore Golden Stick can only run on Ubuntu18.04.

### Version dependency

The MindSpore Golden Stick depends on the MindSpore training and inference framework, please refer the table below and [MindSpore Installation Guide](https://mindspore.cn/install) to install the corresponding MindSpore verision:

| MindSpore Golden Stick Version |                            Branch                            | MindSpore version |
| :-----------------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

After MindSpore is installed, you can use pip or source code build for MindSpore Golden Stick installation.

### Installing from pip command

If you use the pip command, please download the whl package from [MindSpore Golden Stick](https://www.mindspore.cn/versions/en) page and install it.

```shell
pip install  https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore_version}/golden_stick/any/mindspore_rl-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Installing whl package will download MindSpore Golden Stick dependencies automatically (detail of dependencies is shown in requirement.txt),  other dependencies should install manually.
> - `{MindSpore_version}` stands for the version of MindSpore. For the version matching relationship between MindSpore and MindSpore Golden Stick, please refer to [page](https://www.mindspore.cn/versions).
> - `{ms_version}` stands for the version of MindSpore Golden Stick. For example, if you would like to download version 0.1.0, you should fill 1.8.0 in `{MindSpore_version}` and fill 0.1.0 in `{mg_version}`.

### Installing from source code

Download [source code](https://gitee.com/mindspore/golden-stick), then enter the `golden-stick` directory.

```shell
bash build.sh
pip install output/mindspore_gs-{mg_version}-py3-none-any.whl
```

`build.sh` is the compiling script in `golden-stick` directory.

### Verification

If you can successfully execute following command, then the installation is completed.

```python
import mindspore_gs
```

## Quick Start

Take [Simulated Quantization (SimQAT)](https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_zh_cn/quantization/simqat.md) as an example for demonstrating how to use MindSpore Golden Stick.

## Documents

### Developer Guide

For more details about the installation guide, tutorials, and APIs, see [MindSpore Golden Stick API Docs].

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) developer communication platform

## Contributions

Welcome to MindSpore contribution.

## License

[Apache License 2.0](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)
