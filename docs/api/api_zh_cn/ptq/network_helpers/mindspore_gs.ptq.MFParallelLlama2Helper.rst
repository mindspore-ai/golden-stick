mindspore_gs.ptq.MFParallelLlama2Helper
============================================================

.. py:class:: mindspore_gs.ptq.MFParallelLlama2Helper(config: MindFormerConfig = None)

    从 `NetworkHelper` 类派生，用于MindFormers框架ParallelLlamaForCasualLM网络的工具类。

    参数：
        - **config** (MindFormerConfig) - 一个 `MindFormerConfig` 对象，表示对应网络的配置项。

    异常：
        - **TypeError** - `config` 数据类型不是 `MindFormerConfig`。

    .. py:method:: create_network(self)

        创建ParallelLlamaForCasualLM网络。

        返回：
            ParallelLlamaForCasualLM类型的网络。

    .. py:method:: get_decoder_layers(self, network: ParallelLlamaForCausalLM)

        获取网络的decoder层。

        参数：
            - **network** (ParallelLlamaForCausalLM) - 要获取decoder层的网络。

        返回：
            一个元组(cell_name, 'cell')的列表，表示获取到的decoder层及其名称。

    .. py:method:: get_linears(self, decoder_layer: ParallelLlamaTransformerLayer)

        获取decoder中的linear层。

        参数：
            - **decoder_layer** (ParallelLlamaTransformerLayer) - 要获取linear层的decoder层。

        返回：
            一个'Cell'列表，表示decoder层的linear层。

    .. py:method:: get_page_attention_mgr(self, decoder_layer: ParallelLlamaTransformerLayer)

        获取decoder中的PageAttentionMgr层。

        参数：
            - **decoder_layer** (ParallelLlamaTransformerLayer) - 要获取PageAttentionMgr层的decoder层。

        返回：
            一个'Cell'列表，表示decoder层的PageAttentionMgr层。

    .. py:method:: analysis_decoder_groups(self, network: ParallelLlamaForCausalLM)

        分析网络中decoder组的信息。

        参数：
            - **network** (ParallelLlamaForCausalLM) - 要分析decoder组信息的网络。
