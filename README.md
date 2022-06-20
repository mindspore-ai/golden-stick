# MindSpore Golden Stick

[查看中文](./README_CN.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [Overview](#overview)
    - Installation
        - [Version dependency](#version-dependency)
        - [Installing from pip command](#installing-from-pip-command)
        - [Installing from source code](#installing-from-source-code)
    - [Quick Start](#quick-start)
    - Documents
        - [Developer Guide](#developer-guide)
    - Community
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

1. Provide user-centric APIs to reduce user learning costs: MindSpore Golden Stick defines an abstract class of algorithms. All algorithm implementations should inherit from this base class. Users can also use the interfaces defined by the base class to directly apply all algorithms without learning the application of each algorithm. Way. This also facilitates the subsequent exploration of combined algorithms or algorithm search optimization based on the algorithm ecology.
2. Provide some infrastructure capabilities to reduce algorithm access costs: MindSpore Golden Stick provides some infrastructure capabilities for algorithm implementation, such as commissioning and network modification. The debugging capabilities mainly include network dumping, layer-by-layer profiling and other capabilities, which are designed to help algorithm developers locate bugs in algorithm implementation and help users find algorithms that meet their needs. The network modification capability refers to the ability to modify the network structure defined by Python through a series of APIs, which aims to allow algorithm developers to focus on the implementation of the algorithm without repeating the wheel for different network definitions.

### Future Roadmap

The current release of MindSpore Golden Stick contains a stable API and provides a linear quantization algorithm, a nonlinear quantization algorithm and a structured pruning algorithm. More algorithms, better network support and debugging capabilities will be provided in the future. Later, with the enrichment of algorithms, the ability to combine algorithms and algorithm search optimization is also planned, so stay tuned.

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

Take [Simulated Quantization (SimQAT)](https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_zh_cn/quantization/quantization.md) as an example for demonstrating how to use MindSpore Golden Stick.

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


