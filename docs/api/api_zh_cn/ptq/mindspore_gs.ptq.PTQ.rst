mindspore_gs.ptq.PTQ
============================================================

.. py:class:: mindspore_gs.ptq.PTQ(config: Union[dict, PTQConfig] = None, layer_policies=None)

    量化算法PTQ的基本实现，支持激活、权重和kvcache的组合量化。

    参数：
        - **config** (:class:`mindspore_gs.ptq.PTQConfig`, 可选) - 用于配置PTQ量化算法，默认值为 ``None``。
        - **layer_policies** (OrderedDict, 可选) - 层的量化策略，默认值为 ``None`` 。`layer_policies` 的key为正则字符串用来匹配层的名称，value为 :class:`mindspore_gs.ptq.PTQConfig`。

    异常：
        - **TypeError** - `config` 在输入不为 ``None`` 时，元素类型不为 PTQConfig。
        - **TypeError** - `layer_policies` 中任一元素不为 ``None`` 时，其类型不为 PTQConfig。
        - **ValueError** - `config` 中的 `mode` 是PTQMode.QUANTIZE时非PYNATIVE模式。
        - **ValueError** - 当act_quant_dtype是int8类型，weight_quant_dtype为None时。
        - **TypeError** - `layer_policies` 的类型不为 OrderedDict。

    .. py:method:: apply(network: Cell, network_helper: NetworkHelper = None, datasets=None, **kwargs)

        将 `network` 中添加伪量化节点，转换成一个伪量化网络。

        参数：
            - **network** (Cell) - 待伪量化的网络。
            - **network_helper** (NetworkHelper) - 网络量化工具，用于解耦算法层和网络框架层。
            - **datasets** (Dataset) - 校准用的数据集。

        返回：
            伪量化后的网络。
                
        异常：
            - **RuntimeError** - 如果当前算法没有有效的初始化。
            - **TypeError** - `network` 不是一个 `Cell` 对象。
            - **ValueError** - 当datasets为空。

    .. py:method:: convert(net_opt: Cell, ckpt_path="")

        将量化网络 `net_opt` 转换为真实量化网络，后续导出用于部署。

        参数：
            - **net_opt** (Cell) - 经过量化算法apply之后的网络。
            - **ckpt_path** (str) - 网络的checkpoint file文件路径，默认值为 ``""``，表示不加载。注意，该参数会在后续版本中被遗弃。

        返回：
            转换后的网络。

        异常：
            - **TypeError** - `net_opt` 数据类型不是Cell。
            - **TypeError** - `ckpt_path` 数据类型不是str。
            - **ValueError** - `ckpt_path` 非空但不是有效路径。

    .. py:method:: summary(network)

        总结并打印网络中每一层的使用的模型压缩策略，如某一个线性层的量化策略为：W8-perchannel-A8-pertensor。

        参数：
            - **network** (Cell) - 待总结的网络。

        异常：
            - **TypeError** - `network` 数据类型不是Cell。
