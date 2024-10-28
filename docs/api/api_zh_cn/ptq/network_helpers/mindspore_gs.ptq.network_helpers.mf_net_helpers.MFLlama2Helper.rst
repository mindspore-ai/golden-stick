mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper
====================================================================

.. py:class:: mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper(config: Union[str, MindFormerConfig] = None)

    从 `NetworkHelper` 类派生，用于MindFormers框架Llama2网络的工具类。

    参数：
        - **config** (MindFormerConfig) - 一个 `MindFormerConfig` 对象，表示对应网络的配置项。

    异常：
        - **TypeError** - `config` 数据类型不是 `MindFormerConfig`。

    .. py:method:: assemble_inputs(input_ids: np.ndarray, **kwargs)

        根据输入的numpy格式的tokens，编译网络推理所需的输入。

        参数：
            - **input_ids** (numpy.ndarray) - 输入的tokens。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个 `mindspore.Tensor` 的列表，表示用于网络推理的输入。
