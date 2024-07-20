mindspore_gs.ptq.RoundToNearest
============================================================

.. py:class:: mindspore_gs.ptq.RoundToNearest(config=None)

    后量化算法的基本实现，通过统计最大最小值实现模型量化。

    参数：
        - **config** (:class:`mindspore_gs.ptq.PTQConfig`, 可选) - 用于配置后训练量化算法，默认值为 ``None``。

    异常：
        - **TypeError** - `config` 在输入不为 ``None`` 时，元素类型不为 PTQConfig。

    .. py:method:: apply(network: Cell, network_helper: NetworkHelper = None)

        将 `network` 中添加伪量化节点，转换成一个伪量化网络。

        参数：
            - **network** (Cell) - 待伪量化的网络。
            - **network_helper** (NetworkHelper) - 网络量化工具，用于解耦算法层和网络框架层。
        
        异常：
            - **RuntimeError** - 如果当前算法没有有效的初始化。

        返回：
            伪量化后的网络。

    .. py:method:: convert(net_opt: Cell, ckpt_path="")

        将量化网络 `net_opt` 转换为真实量化网络，后续导出用于部署。

        参数：
            - **net_opt** (Cell) - 经过量化算法apply之后的网络。
            - **ckpt_path** (str) - 网络的checkpoint file文件路径，默认值为 ``""``，表示不加载。注意，该参数会在后续版本中被遗弃。

        异常：
            - **TypeError** - `net_opt` 数据类型不是Cell。
            - **TypeError** - `ckpt_path` 数据类型不是str。
            - **ValueError** - `ckpt_path` 非空但不是有效路径。

        返回：
            转换后的网络。
