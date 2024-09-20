# MindSpore Golden Stick

[查看中文](./README_CN.md)

<!-- TOC -->

- MindSpore Golden Stick
    - [Overview](#overview)
    - [Design Features](#design-features)
    - [General Process of Applying the MindSpore Golden Stick](#general-process-of-applying-the-mindspore-golden-stick)
    - [Documents](#documents)
        - [Installation](#installation)
        - [Quick Start](#quick-start)
        - [Compression Algorithm](#compression-algorithm)
        - [Model Deployment](#model-deployment)
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

![MindSpore_GS_Architecture](docs/images/en/golden-stick-arch.png)

> The architecture diagram is the overall picture of MindSpore Golden Stick, which includes the features that have been implemented in the current version and the capabilities planned in RoadMap. Please refer to release notes for available features in current version.

## Design Features

In addition to providing rich model compression algorithms, an important design concept of MindSpore Golden Stick is try to provide users with the most unified and concise experience for a wide variety of model compression algorithms in the industry, and reduce the cost of algorithm application for users. MindSpore Golden Stick implements this philosophy through two initiatives:

1. Unified algorithm interface design to reduce user application costs:

   There are many types of model compression algorithms, such as quantization-aware training algorithms, pruning algorithms, matrix decomposition algorithms, knowledge distillation algorithms, etc. In each type of compression algorithm, there are also various specific algorithms, such as LSQ and PACT, which are both quantization-aware training algorithms. Different algorithms are often applied in different ways, which increases the learning cost for users to apply algorithms. MindSpore Golden Stick sorts out and abstracts the algorithm application process, and provides a set of unified algorithm application interfaces to minimize the learning cost of algorithm application. At the same time, this also facilitates the exploration of advanced technologies such as AMC, NAS and HAQ based on the algorithm ecology.

2. Provide front-end network modification capabilities to reduce algorithm development costs：

   Model compression algorithms are often designed or optimized for specific network structures. For example, perceptual quantization algorithms often insert fake-quantization nodes on the Conv2d, Conv2d + BatchNorm2d, or Conv2d + BatchNorm2d + Relu structures in the network. MindSpore Golden Stick provides the ability to modify the front-end network through API. Based on this ability, algorithm developers can formulate general network transform rules to implement the algorithm logic without needing to implement the algorithm logic for each specific network. In addition, MindSpore Golden Stick also provides some debugging capabilities, including visualization tool, profiler tool, summary tool, aiming to help algorithm developers improve development and research efficiency, and help users find algorithms that meet their needs.

## General Process of Applying the MindSpore Golden Stick

![workflow](docs/images/en/workflow.png)

1. Compress Phase

    Taking the quantization algorithm as an example, the compression phase mainly includes transforming the network into a fake-quantized network, quantization retraining or calibration, quantizing parameter statistics, quantizing weights, and transforming the network into a real quantized network.

2. Deplyment Phase

    The deployment phase is the process of inferring the compressed network in the deployment environment. Since MindSpore does not support serialization of the front-end network, the deployment also needs to call the corresponding algorithm interface to transform the network to load the compressed checkpoint file. The flow after loading the compressed checkpoint file is the same as the normal inference process.

> - For details about how to apply the MindSpore Golden Stick, see the detailed description and sample code in each algorithm section.
> - For details about the "ms.export" step in the process, see [Exporting MINDIR Model](https://www.mindspore.cn/tutorials/en/master/beginner/save_load.html#saving-and-loading-mindir).
> - For details about the "MindSpore infer" step in the process, see [MindSpore Inference Runtime](https://mindspore.cn/docs/en/master/model_infer/ms_infer/overview.html).

## Documents

### Installation

Please refer to [MindSpore Golden Stick Installation](docs/docs/docs_en/install.md).

### Quick Start

Take [Simulated Quantization (SimQAT)](https://gitee.com/mindspore/docs/blob/master/docs/golden_stick/docs/source_zh_cn/quantization/simqat.md) as an example for demonstrating how to use MindSpore Golden Stick.

### Compression Algorithm

| type |                             link                                  |
| :---------------------: | :-----------------------------------------------------------------: |
| API         | [API doc](https://www.mindspore.cn/golden_stick/docs/en/master) |
| AutoCompress   | TBD |
| Post-Training Quantization   | [PTQ](mindspore_gs/ptq/ptq/README.md) [RoundToNearest](mindspore_gs/ptq/round_to_nearest/README.md) |
| [Quant-Aware Quantization](mindspore_gs/quantization/README.md) | [SimQAT](mindspore_gs/quantization/simulated_quantization/README.md) [SLB](mindspore_gs/quantization/slb/README.md) |
| [Pruner](mindspore_gs/pruner/README.md) | [SCOP](mindspore_gs/pruner/scop/README.md) [uni_pruning(demo)](mindspore_gs/pruner/uni_pruning/README.md) [LRP(demo)](mindspore_gs/pruner/heads/lrp/README.md)  |
| Others | [Ghost](mindspore_gs/ghost/README.md)  |

### Model Deployment

Please refer to [MindSpore Golden Stick Model Deployment](docs/docs/docs_en/deployment/overview.md)。

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) developer communication platform

## Contributions

Welcome to MindSpore contribution.

## License

[Apache License 2.0](https://gitee.com/mindspore/golden-stick/blob/master/LICENSE)
