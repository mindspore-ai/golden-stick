# 架构设计

![金箍棒架构图](images/golden-stick-arch.png)

架构图是MindSpore Golden Stick的全貌，其中包含了当前已经实现的功能以及规划在RoadMap中能力。具体开放的功能可以参考对应版本的ReleaseNotes。

## 设计思路

MindSpore Golden Stick除了提供丰富的模型压缩算法外，一个重要的设计理念是针对业界种类繁多的模型压缩算法，提供给用户一个尽可能统一且简洁的体验，降低用户的算法应用成本。MindSpore Golden Stick通过两个举措来实现该理念：

1. 统一的算法接口设计，降低用户应用成本

   模型压缩算法种类繁多，有如量化感知训练算法、剪枝算法、矩阵分解算法、知识蒸馏算法等；在每类压缩算法中，还有会各种具体的算法，比如LSQ、PACT都是量化感知训练算法。不同算法的应用方式往往各不相同，这增加了用户应用算法的学习成本。MindSpore Golden Stick对算法应用流程做了梳理和抽象，提供了一套统一的算法应用接口，最大程度缩减算法应用的学习成本。同时这也方便了后续在算法生态的基础上，做一些AMC、NAS、HAQ等高阶技术的探索。

2. 提供前端网络修改能力，降低算法接入成本

   模型压缩算法往往会针对特定的网络结构做设计或者优化，如感知量化算法往往在网络中的Conv2d、Conv2d + BatchNorm2d或者Conv2d + BatchNorm2d + Relu结构上插入伪量化节点。MindSpore Golden Stick提供了通过接口修改前端网络的能力，算法开发者可以基于此能力制定通用的改图规则去实现算法逻辑，而不需要对每个特定的网络都实现一遍算法逻辑算法。此外MindSpore Golden Stick还会提供了一些调测能力，包括网络dump、逐层profiling、算法效果分析、可视化等能力，旨在帮助算法开发者提升开发和研究效率，帮助用户寻找契合于自己需求的算法。

## 应用MindSpore Golden Stick算法的一般流程

![金箍棒流程图](images/workflow.png)

1. 压缩阶段

    压缩阶段是指使用MindSpore Golden Stick算法对网络进行压缩的过程，以量化算法为例，压缩阶段主要包含改造网络为伪量化网络、量化重训或者校正、量化参数统计、量化权重、改造网络为真实量化网络。

2. 部署阶段

    部署阶段是将压缩后的网络在部署环境进行推理的过程，由于MindSpore暂不支持将前端网络进行序列化，所以部署同样需要调用对应的算法接口对网络进行改造，以加载压缩后的checkpoint文件。加载完压缩的checkpoint文件以后的流程和一般的推理流程无异。

> - 应用MindSpore Golden Stick算法的细节，可以在每个算法章节中找到详细说明和示例代码。
> - 流程中的"ms.export"步骤可以参考[导出mindir格式文件](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html#保存和加载mindir)章节。
> - 流程中的"昇思推理优化工具和运行时"步骤可以参考[昇思推理](https://www.mindspore.cn/tutorials/zh-CN/master/model_infer/ms_infer/ms_infer_model_infer.html)章节。
