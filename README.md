# MindSpore Golden Stick

[查看中文](./README_CN.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [Overview](#overview)
    - [Design Features](#design-features)
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

MindSpore Golden Stick is an open source model compression algorithm set, which provides a user interface allowing users to apply model compression algorithms such as quantization and pruning in a unified and convenient manner.

MindSpore Golden Stick also provides the infrastructure for algorithm developers to modify network definitions. It abstracts a layer of IR between algorithms and network definitions, shielding algorithm developers from specific network definitions so that they can focus on the development of algorithm logic.

![MindSpore_GS_Architecture](docs/golden-stick-arch.png)

## Design Features

1. Provide user-centric APIs to reduce user learning costs:

   There are many types of model compression algorithms, such as quantization-aware training algorithms, pruning algorithms, matrix decomposition algorithms, knowledge distillation algorithms, etc. In each type of compression algorithm, there are also various specific algorithms, such as LSQ and PACT, which are both quantization-aware training algorithms. Different algorithms usually have different applying method, which increases the learning cost for users to apply algorithms. MindSpore Golden Stick sorts out and abstracts the algorithm application process, and provides a unified set of algorithm application interfaces to minimize the learning cost while applying model compression algorithm. At the same time, this feature also facilitates the exploration of technologies such as AMC (Automatic Model Compression Technology) and NAS (Network Structure Search Technology) based on the algorithm ecology.

2. Provide some infrastructure capabilities to reduce algorithm access costs:

   Model compression algorithms are often designed or optimized for specific network structures, but rarely pay attention to specific network definitions. MindSpore Golden Stick provides the ability to modify the front-end network definition through the API, allowing algorithm developers to focus on the implementation of the algorithm without reinventing the wheel for different network definitions. In addition, MindSpore Golden Stick will also provide some debugging capabilities, including network dump, layer-by-layer profiling, algorithm effect analysis, visualization and other capabilities, aiming to help algorithm developers improve development and research efficiency, and help users find algorithms that meet their needs.

### Future Roadmap

The current release version of MindSpore Golden Stick provides a stable API and provides a linear quantization algorithm, a nonlinear quantization algorithm and a structured pruning algorithm. More algorithms and better network support will be provided in the future version, and debugging capabilities will also be provided in subsequent versions. With the enrichment of algorithms in the future, MindSpore Golden Stick will also explore capabilities such as AMC, HAQ(Hardware-Aware Automated Quantization), NAS, etc., so stay tuned.

## Installation

MindSpore Golden Stick depends on the MindSpore training and inference framework. Therefore, please first install [MindSpore](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) following the instruction on the official website, then install MindSpore Golden Stick. You can install from `pip` or source code.

### Version dependency

Due the dependency between MindSpore Golden Stick and MindSpore, please follow the table below and install the corresponding MindSpore verision from [MindSpore download page](https://www.mindspore.cn/versions/en).

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore-Version}/MindSpore/cpu/ubuntu_x86/mindspore-{MindSpore-Version}-cp37-cp37m-linux_x86_64.whl
```

| MindSpore Golden Stick Version |                            Branch                            | MindSpore version |
| :-----------------------------: | :----------------------------------------------------------: | :-------: |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/) |   1.8.0   |

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
pip install output/mindspore_rl-{mg_version}-py3-none-any.whl
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
