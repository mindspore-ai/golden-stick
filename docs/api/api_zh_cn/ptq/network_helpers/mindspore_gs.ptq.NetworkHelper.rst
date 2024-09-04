mindspore_gs.ptq.NetworkHelper
============================================================

.. py:class:: mindspore_gs.ptq.NetworkHelper()

    工具类，用于解耦算法层和网络框架层，使算法实现不依赖于具体的框架。

    .. py:method:: analysis_decoder_groups(self, network: Cell)

        分析网络中decoder组的信息。

        参数：
            - **network** (Cell) - 要分析decoder组信息的网络。

    .. py:method:: assemble_inputs(input_ids: np.ndarray, **kwargs)

        根据输入的tokens，编译网络推理所需的输入。

        参数：
            - **input_ids** (numpy.ndarray) - 输入的tokens。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个 `mindspore.Tensor` 的列表，表示用于网络推理的输入。

    .. py:method:: create_network()

        创建网络。

        返回：
            创建的网络。

    .. py:method:: create_tokenizer(**kwargs)

        获取网络的分词器。

        参数：
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个对象，表示网络分词器。

    .. py:method:: generate(network: Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs)

        对网络进行自递归式推理，生成一系列tokens。

        参数：
            - **network** (Cell) - 进行自递归生成的网络。
            - **input_ids** (numpy.ndarray) - 用于生成的输入tokens。
            - **max_new_tokens** (int) - 最长生成长度，默认值为 ``1``。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个列表，表示生成的tokens。

    .. py:method:: get_decoder_layers(self, network: Cell)

        获取网络的decoder层。

        参数：
            - **network** (Cell) - 要获取decoder层的网络。

        返回：
            一个元组的列表，表示获取到的decoder层及其名称。

    .. py:method:: get_linears(self, decoder_layer: Cell)

        获取decoder中的所有linear层。

        参数：
            - **decoder_layer** (Cell) - 要获取linear层的decoder层。

        返回：
            一个列表，表示获取到的linears层。

    .. py:method:: get_page_attention_mgr(self, decoder_layer: Cell)

        获取decoder中的所有page_attention_mgr层。

        参数：
            - **decoder_layer** (Cell) - 要获取page_attention_mgr层的decoder层。

        返回：
            一个列表，表示获取到的page_attention_mgr层。

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
