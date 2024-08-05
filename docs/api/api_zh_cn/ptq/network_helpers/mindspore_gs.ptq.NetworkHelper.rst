mindspore_gs.ptq.NetworkHelper
============================================================

.. py:class:: mindspore_gs.ptq.NetworkHelper()

    工具类，用于解耦算法层和网络框架层，使算法实现不依赖于具体的框架。

    .. py:method:: assemble_inputs(input_ids: np.ndarray, **kwargs)

        根据输入的tokens，编译网络推理所需的输入。

        参数：
            - **input_ids** (numpy.ndarray) - 输入的tokens。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个 `mindspore.Tensor` 的列表，表示用于网络推理的输入。

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
            - **max_new_tokens** (int) - 最长生成长度。
            - **kwargs** (Dict) - 用于子类可扩展入参。

        返回：
            一个列表，表示生成的tokens。

    .. py:method:: get_spec(self, name: str)

        获取网络的规格，比如batch_size、seq_length等。

        参数：
            - **name** (str) - 要获取的规格名称。

        返回：
            一个对象，表示获取到的网络规格。

    .. py:method:: get_decoder_layers(self, network: Cell)

        获取网络的decoder层。

        参数：
            - **network** (Cell) - 要获取decoder层的网络。

        返回：
            一个列表，表示获取到的decoder层。

    .. py:method:: get_linears(self, decoder: Cell)

        获取decoder中的所有linear层。

        参数：
            - **decoder** (Cell) - 要获取linear层的decoder层。

        返回：
            一个列表，表示获取到的linears层。

    .. py:method:: offload_embedding(self, network: Cell)

        释放网络中embedding层tensor所占用的显存。

        参数：
            - **network** (Cell) - 要释放显存的网络。
