mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper
====================================================================

.. py:class:: mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper(config: MindFormerConfig = None)

    从 `NetworkHelper` 类派生，用于MindFormers框架Llama2网络的工具类。

    参数：
        - **config** (MindFormerConfig) - 一个 `MindFormerConfig` 对象，表示对应网络的配置项。

    异常：
        - **TypeError** - `config` 数据类型不是 `MindFormerConfig`。

    .. py:method:: analysis_decoder_groups(self, network: Cell)

        分析网络中decoder组的信息。

        参数：
            - **network** (Cell) - 要分析decoder组信息的网络。

    .. py:method:: assemble_inputs(input_ids: np.ndarray, **kwargs)

        根据输入的numpy格式的tokens，编译网络推理所需的输入。

        参数：
            - **input_ids** (numpy.ndarray) - 输入的tokens。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个 `mindspore.Tensor` 的列表，表示用于网络推理的输入。

    .. py:method:: get_pre_layer(self, linear_name: str)

        通过当前linear层的名称，获取前一层的信息。

        参数：
            - **linear_name** (str) - linear层名称。
        
        返回：
            一个字典，表示获取到的前一层layer的信息，包含了layer名称、layer和类型。

    .. py:method:: get_spec(self, name: str)

        获取网络的规格，比如batch_size、seq_length等。

        参数：
            - **name** (str) - 要获取的规格名称。

        返回：
            一个对象，表示获取到的网络规格。
