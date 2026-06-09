MindSpore Golden Stick 文档
=========================================

MindSpore Golden Stick（金箍棒）是MindSpore团队和华为诺亚团队联合设计开发的一个模型压缩工具。我们有两大目标：

1. 我们希望构建MindSpore开源生态的模型压缩能力，并提供简洁易用的接口方便用户提升MindSpore网络的部署效率；
2. 我们希望屏蔽框架和硬件的复杂性，为模型压缩算法提供扩展性良好的基础能力。

MindSpore Golden Stick基于MindSpore内置的模型压缩技术和组件化设计，具备如下特点：

- **SoTA算法：** 金箍棒中的模型压缩算法主要有两大来源，一个是业界的SoTA的算法，我们会持续在MindSpore生态上跟进；另一个是华为算法团队提供的创新性算法；
- **接口易用：** 金箍棒提供类Transformers的接口，并且支持直接将Hugging Face社区的权重进行压缩，输出的权重同样符合Hugging Face社区权重格式；
- **分层解耦：** 金箍棒致力于构建易用的算法预研平台，我们对框架做了分层和模块化设计，一方面屏蔽框架和硬件的复杂性，另一方面方便算法工程师在算法不同层级快速进行创新和实验；
- **硬件适配：** 支持在昇腾硬件上将Hugging Face的权重进行量化，并在vllm-MindSpore Plugin或MindSpore Transformers上进行部署。

用户可以参阅 `架构设计 <design.html>`_ ，快速了解MindSpore Golden Stick的系统架构。

如果您对MindSpore Golden Stick有任何建议，请通过 `issue <https://atomgit.com/mindspore/golden-stick/issues>`_ 与我们联系，我们将及时处理。

使用MindSpore Golden Stick进行模型压缩
-----------------------------------------------------

MindSpore Golden Stick提供了统一的模型压缩接口，支持训练后量化（PTQ）、量化感知训练（QAT）、模型剪枝等多种压缩技术，可以通过以下说明文档进行学习：

当前金箍棒的工作主要聚焦在LLM网络和多模态理解网络的压缩上，主要使用训练后量化技术。QAT算法、剪枝算法主要针对CV网络，技术已经相对陈旧，当前已经不再演进和维护。后续我们也可能规划一些针对LLM网络的QAT或者剪枝算法，也欢迎大家在社区贡献代码或者提交需求。

- `安装指南 <install.html>`_
- `Transformers-Like API <ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html>`_
- `后量化配置接口 <ptq/mindspore_gs.ptq.PTQConfig.html>`_
- `训练后量化指南 <ptq/ptq.html>`_
- `量化感知训练指南 <quantization/overview.html>`_
- `模型剪枝指南 <pruner/overview.html>`_

代码仓地址： <https://atomgit.com/mindspore/golden-stick>

MindSpore Golden Stick支持算法列表
-----------------------------------------------------

- 训练后量化（PTQ）：

  - `PTQ概述 <ptq/overview.html>`_

    训练后量化的基础原理和常见方案。

  - `OutlierSuppressionLite <ptq/ptq.html#outliersuppressionlite%E7%AE%97%E6%B3%95>`_

    华为泰勒团队与MindSpore团队联合开发的创新算法，提供更高精度的A8W8量化能力。

  - `A8W4混合精度量化 <ptq/ptq.html#%E9%87%91%E7%AE%8D%E6%A3%92%E6%94%AF%E6%8C%81%E7%BB%84%E5%90%88%E9%87%8F%E5%8C%96>`_

    结合OutlierSuppressionLite和GPTQ算法，实现层级混合精度量化。

  - `RoundToNearest量化 <ptq/ptq.html#roundtonearest%E7%AE%97%E6%B3%95>`_

    支持A16W8量化，提供基础的权重量化能力。

  - `SmoothQuant量化 <ptq/ptq.html#smoothquant%E7%AE%97%E6%B3%95>`_

    支持A8W8量化，通过平滑激活值分布提升量化精度。

  - `AWQ量化 <ptq/ptq.html#awq%E7%AE%97%E6%B3%95>`_

    支持A16W4量化，通过激活感知权重量化提升低比特量化效果。

  - `GPTQ量化 <ptq/ptq.html#gptq%E7%AE%97%E6%B3%95>`_

    支持A16W4量化，基于梯度的训练后量化算法。

  - `动态量化 <ptq/ptq.html#%E5%8A%A8%E6%80%81%E9%87%8F%E5%8C%96%E7%AE%97%E6%B3%95>`_

    支持per-token动态量化，在推理过程中实时计算量化参数。

- 量化感知训练（QAT）：

  - `QAT概述 <quantization/overview.html>`_

    量化感知训练的基础原理和常见方案。

  - `模拟量化训练 <quantization/simulated_quantization.html>`_

    在训练过程中模拟量化效果，提升量化模型精度。

  - `SLB量化训练 <quantization/slb.html>`_

    Searching for Low-Bit量化训练算法。

- 剪枝：

  - `剪枝概述 <pruner/overview.html>`_

    剪枝的基础原理和常见方案。

  - `SCOP剪枝 <pruner/scop.html>`_

    支持通道级别的结构化剪枝，减少模型参数量。

- 部署集成

  - `vLLM-MindSpore Plugin集成 <https://atomgit.com/mindspore/vllm-mindspore>`_
  - `MindSpore Transformers集成 <https://atomgit.com/mindspore/mindformers>`_

- 贡献指南

  - `MindSpore Golden Stick贡献指南 <CONTRIBUTING.html>`_

- RELEASE NOTES

  - `MindSpore Golden Stick Release Notes <RELEASE.html>`_

------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: 架构设计
   :hidden:

   design

.. toctree::
   :maxdepth: 1
   :caption: 安装部署
   :hidden:

   install

.. toctree::
   :maxdepth: 1
   :caption: 训练后量化
   :hidden:

   ptq/overview
   ptq/ptq
   ptq/round_to_nearest

.. toctree::
   :maxdepth: 1
   :caption: 量化感知训练
   :hidden:

   quantization/overview
   quantization/simulated_quantization
   quantization/slb

.. toctree::
   :maxdepth: 1
   :caption: 剪枝
   :hidden:

   pruner/overview
   pruner/scop

.. toctree::
   :maxdepth: 1
   :caption: API参考
   :hidden:

   mindspore_gs
   mindspore_gs.common
   mindspore_gs.ptq
   mindspore_gs.quantization
   mindspore_gs.pruner

.. toctree::
   :maxdepth: 1
   :caption: 贡献指南
   :hidden:

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE
