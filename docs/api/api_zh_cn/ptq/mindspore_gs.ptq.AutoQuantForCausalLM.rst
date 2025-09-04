mindspore_gs.ptq.AutoQuantForCausalLM
============================================================

.. py:class:: mindspore_gs.ptq.AutoQuantForCausalLM

    自动模型量化类，根据预训练模型路径自动选择合适的量化模型实现。

    该类提供了因果语言模型的自动模型检测和选择功能。它使用注册机制根据预训练模型配置自动识别并实例化适当的量化模型实现。

    该类实现了一个工厂模式，扫描所有已注册的模型中心，并尝试从每个中心创建模型实例，直到找到成功匹配的模型。

    .. py:method:: from_pretrained(pretained) -> BaseQuantForCausalLM
        :staticmethod:

        根据预训练模型路径创建量化模型实例。

        此方法自动从提供的预训练模型路径中检测模型类型，并选择适当的量化模型实现。它会遍历所有已注册的模型中心，并尝试从每个中心创建模型实例。

        参数：
            - **pretained** (str) - 预训练模型的路径或标识符。这可以是模型配置文件的本地文件路径，或者是系统识别的模型标识符。

        返回：
            `BaseQuantForCausalLM` - 继承自BaseQuantForCausalLM的量化模型实例。具体类型取决于检测到的模型框架和配置。
