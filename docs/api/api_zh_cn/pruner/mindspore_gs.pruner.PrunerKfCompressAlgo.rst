mindspore_gs.pruner.PrunerKfCompressAlgo
========================================

.. py:class:: mindspore_gs.pruner.PrunerKfCompressAlgo(config=None)

    `PrunerKfCompressAlgo` 是CompAlgo的子类，实现了SCOP算法中利用高仿数据来学习发现冗余卷积核。

    .. note::
        - 针对入参 `config` ，目前 `PrunerKfCompressAlgo` 是没有可选的配置项，但为了兼容性， `config` 被保留，在初始化时以空字典代替。如 `kf_pruning = PrunerKfCompressAlgo({})` 。

    参数：
        - **config** (dict) - 算法配置参数。

    .. py:method:: apply(network, **kwargs)

        将网络变成Konckoff网络。

        参数：
            - **network** (Cell) - 原始待剪枝网络。
            - **kwargs** (Dict) - 用于子类的可扩展入参。

        返回：
            返回变换后的Konckoff网络。
        
        异常：
            - **TypeError** - `network` 不是Cell。

    .. py:method:: callbacks(*args, **kwargs)

        定义SCOP剪枝算法特有的callbacks即生成高仿数据的callback。

        返回：
            SCOP剪枝算法的callbacks列表。
