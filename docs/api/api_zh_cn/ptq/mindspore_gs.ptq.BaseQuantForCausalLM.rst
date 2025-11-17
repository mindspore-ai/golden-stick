mindspore_gs.ptq.BaseQuantForCausalLM
============================================================

.. py:class:: mindspore_gs.ptq.BaseQuantForCausalLM

    因果语言模型量化基类，定义了量化模型的标准接口。

    该基类为所有量化因果语言模型实现定义了标准接口。它提供了所有派生类必须实现的基本结构和必需方法。

    该类实现了一个注册机制，允许不同的模型框架注册它们的实现。这使得AutoQuantForCausalLM能够提供自动模型检测和选择功能。

    .. py:method:: calibrate(ptq_config, layers_policy, datasets, **kwargs)

        模型校准和量化。

        这是一个抽象方法，必须由派生类实现。它应该使用校准数据集处理模型校准过程，并根据提供的配置应用量化。

        参数：
            - **ptq_config** (PTQConfig) - 训练后量化的配置。
            - **layers_policy** (dict) - 不同层量化策略的策略。
            - **datasets** (Dataset) - 用于量化的校准数据集。
            - **kwargs** (dict) - 其他关键字参数。

        异常：
            - **NotImplementedError** - 子类必须实现此方法。

    .. py:method:: fake_quant(ptq_config, layers_policy, quant_safetensors_path="")

        对模型应用伪量化。

        该方法对模型应用伪量化，这对于验证量化效果而不实际转换为整数操作非常有用。

        参数：
            - **ptq_config** (PTQConfig) - 训练后量化的配置。
            - **layers_policy** (dict) - 不同层量化策略的策略。
            - **quant_safetensors_path** (str, 可选) - 量化safetensors的路径。默认为 ``""``。

        异常：
            - **NotImplementedError** - 子类必须实现此方法。
    
    .. py:method:: forward(input_ids, max_new_tokens=1)

        模型前向推理。

        这是一个抽象方法，必须由派生类实现。它应该处理模型推理的前向传递逻辑。

        参数：
            - **input_ids** (Tensor) - 模型的输入token ID。
            - **max_new_tokens** (int, 可选) - 最大生成token数，默认为 ``1``。

        返回：
            前向传递结果。

        异常：
            - **NotImplementedError** - 子类必须实现此方法。

    .. py:method:: from_pretrained(**kwargs)
        :classmethod:

        从预训练模型创建量化实例。

        这是一个抽象方法，必须由派生类实现。它应该处理加载预训练模型权重和配置。

        参数：
            - **kwargs** (dict) - 创建模型所需的任意关键字参数。

        返回：
            `BaseQuantForCausalLM` - 量化模型实例。

        异常：
            - **NotImplementedError** - 子类必须实现此方法。

    .. py:method:: save_quantized(save_path)

        保存量化模型到磁盘。

        这是一个抽象方法，必须由派生类实现。它应该处理保存量化模型权重和配置。

        参数：
            - **save_path** (str) - 应保存量化模型的路径。

        异常：
            - **NotImplementedError** - 子类必须实现此方法。
