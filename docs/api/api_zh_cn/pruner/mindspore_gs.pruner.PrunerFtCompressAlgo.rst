mindspore_gs.pruner.PrunerFtCompressAlgo
========================================

.. py:class:: mindspore_gs.pruner.PrunerFtCompressAlgo(config=None)

    `PrunerFtCompressAlgo` 是CompAlgo的子类，实现了删除冗余卷积核并对网络进行完整训练的能力。
    
    参数：
        - **config** (dict) - 以字典的形式存放用于剪枝训练的配置，下面列出了受支持的属性：
          
          - **prune_rate** (float) - 值的取值范围是[0.0, 1.0)。
    
    异常：
        - **TypeError** - `prune_rate` 的数据类型不是 `float` 。
        - **ValueError** - `prune_rate` 小于0或者大于等于1。

    .. py:method:: apply(network, **kwargs)

        将Konckoff网络变为剪枝后的网络。

        参数：
            - **network** (Cell) - Konckoff网络。
            - **kwargs** (Dict) - 用于子类的可扩展入参。

        返回：
            返回剪枝后的网络。

        异常：
            - **TypeError** - `network` 不是Cell。

    .. py:method:: set_prune_rate(prune_rate: float)

        设置剪枝率。

        参数：
            - **prune_rate** (float) - 剪掉网络的大小。

        异常：
            - **TypeError** - `prune_rate` 不是float。
            - **ValueError** - `prune_rate` 小于0或者大于等于1。