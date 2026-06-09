MindSpore Golden Stick Documentation
=======================================

MindSpore Golden Stick is a model compression toolkit jointly designed and developed by the MindSpore team and Huawei Noah's Ark Lab. We have two major goals:

1. Build model compression capabilities for the MindSpore open-source ecosystem and provide simple, easy-to-use interfaces to improve deployment efficiency of MindSpore networks.
2. Shield the complexity of frameworks and hardware while offering extensible foundational capabilities for model compression algorithms.

Based on MindSpore's built-in compression technologies and a componentized design, MindSpore Golden Stick features:

- **SoTA Algorithms:** The model compression algorithms in Golden Stick mainly come from two sources: one is the state-of-the-art algorithms from the industry, which we continuously follow up on in the MindSpore ecosystem; the other is innovative algorithms provided by Huawei's algorithm teams.
- **Easy-to-use Interface:** Golden Stick provides Transformers-like interfaces and supports direct compression of Hugging Face community weights, with output weights that also conform to the Hugging Face community weight format.
- **Layered Decoupling:** Golden Stick is committed to building an easy-to-use algorithm research platform. We have designed the framework with layered and modular architecture, which on one hand shields the complexity of frameworks and hardware, and on the other hand facilitates algorithm engineers to quickly innovate and experiment at different levels of algorithms.
- **Hardware adaptation:** Supports quantizing Hugging Face weights on Ascend hardware and deployment via the vLLM-MindSpore Plugin or MindSpore Transformers.

You can refer to the `Architecture Design <design.html>`_ to quickly understand the system architecture of MindSpore Golden Stick.

If you have any suggestions for MindSpore Golden Stick, please contact us via `issues <https://atomgit.com/mindspore/golden-stick/issues>`_, and we will respond promptly.

Using MindSpore Golden Stick for Model Compression
-----------------------------------------------------

MindSpore Golden Stick provides unified model compression interfaces and supports multiple compression techniques such as Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and model pruning. You can learn more from the following documentation:

Currently, Golden Stick focuses primarily on compressing LLMs and multimodal understanding models, mainly with PTQ. QAT and pruning algorithms were originally designed for CV models and are no longer under active evolution and maintenance. We may plan QAT or pruning algorithms for LLMs in the future; contributions and feature requests are welcome in the community.

- `Installation Guide <install.html>`_
- `Transformers-Like API <ptq/mindspore_gs.ptq.AutoQuantForCausalLM.html>`_
- `PTQ Configuration API <ptq/mindspore_gs.ptq.PTQConfig.html>`_
- `Post-Training Quantization Guide <ptq/ptq.html>`_
- `Quantization-Aware Training Guide <quantization/overview.html>`_
- `Model Pruning Guide <pruner/overview.html>`_

Repository: <https://atomgit.com/mindspore/golden-stick>

Supported Algorithms in MindSpore Golden Stick
--------------------------------------------------

- Post-Training Quantization (PTQ):

  - `PTQ Overview <ptq/overview.html>`_

    Basic principles and common approaches of post-training quantization.

  - `OutlierSuppressionLite <ptq/ptq.html#outliersuppressionlite-algorithm>`_

    An innovative algorithm jointly developed by Huawei Taylor team and the MindSpore team, providing higher-accuracy A8W8 quantization.

  - `A8W4 Mixed-Precision Quantization <ptq/ptq.html#golden-stick-supports-combination-quantization>`_

    Combines OutlierSuppressionLite and GPTQ to achieve layer-wise mixed precision quantization.

  - `RoundToNearest Quantization <ptq/ptq.html#roundtonearest-algorithm>`_

    Supports A16W8 quantization and provides fundamental weight quantization capability.

  - `SmoothQuant <ptq/ptq.html#smoothquant-algorithm>`_

    Supports A8W8 quantization and improves accuracy by smoothing activation distributions.

  - `AWQ <ptq/ptq.html#awq-algorithm>`_

    Supports A16W4 quantization with activation-aware weight quantization to improve low-bit performance.

  - `GPTQ <ptq/ptq.html#gptq-algorithm>`_

    Supports A16W4 quantization with a gradient-based post-training quantization method.

  - `Dynamic Quantization <ptq/ptq.html#dynamic-quantization-algorithm>`_

    Supports per-token dynamic quantization, computing quantization parameters in real time during inference.

- Quantization-Aware Training (QAT):

  - `QAT Overview <quantization/overview.html>`_

    Basic principles and common approaches of quantization-aware training.

  - `Simulated Quantization Training <quantization/simulated_quantization.html>`_

    Simulates quantization effects during training to improve the accuracy of quantized models.

  - `SLB Quantization Training <quantization/slb.html>`_

    Searching for Low-Bit quantization-aware training algorithm.

- Pruning:

  - `Pruning Overview <pruner/overview.html>`_

    Basic principles and common approaches of pruning.

  - `SCOP Pruning <pruner/scop.html>`_

    Supports channel-level structured pruning to reduce the number of model parameters.

- Deployment Integration

  - `vLLM-MindSpore Plugin Integration <https://atomgit.com/mindspore/vllm-mindspore>`_
  - `MindSpore Transformers Integration <https://atomgit.com/mindspore/mindformers>`_

- Contribution Guide

  - `MindSpore Golden Stick Contribution Guide <CONTRIBUTING.html>`_

- RELEASE NOTES

  - `MindSpore Golden Stick Release Notes <RELEASE.html>`_

------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Architecture Design
   :hidden:

   design

.. toctree::
   :maxdepth: 1
   :caption: Installation and Deployment
   :hidden:

   install

.. toctree::
   :maxdepth: 1
   :caption: Post-Training Quantization
   :hidden:

   ptq/overview
   ptq/ptq
   ptq/round_to_nearest

.. toctree::
   :maxdepth: 1
   :caption: Quantization-Aware Training
   :hidden:

   quantization/overview
   quantization/simulated_quantization
   quantization/slb

.. toctree::
   :maxdepth: 1
   :caption: Pruner
   :hidden:

   pruner/overview
   pruner/scop

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   mindspore_gs
   mindspore_gs.common
   mindspore_gs.ptq
   mindspore_gs.quantization
   mindspore_gs.pruner

.. toctree::
   :maxdepth: 1
   :caption: Contribution Guide
   :hidden:

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES
   :hidden:

   RELEASE
