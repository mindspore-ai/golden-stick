mindspore_gs.ptq.network_helpers.mf_net_helpers.MFParallelLlama2Helper
====================================================================================================

.. py:class:: mindspore_gs.ptq.network_helpers.mf_net_helpers.MFParallelLlama2Helper(config: Union[str, MindFormerConfig] = None)

    从 `NetworkHelper` 类派生，用于MindFormers框架ParallelLlamaForCasualLM网络的工具类。

    参数：
        - **config** (MindFormerConfig) - 一个 `MindFormerConfig` 对象，表示对应网络的配置项。

    异常：
        - **TypeError** - `config` 数据类型不是 `MindFormerConfig`。

    .. py:method:: create_network()

        创建ParallelLlamaForCasualLM网络。

        返回：
            ParallelLlamaForCasualLM类型的网络。
