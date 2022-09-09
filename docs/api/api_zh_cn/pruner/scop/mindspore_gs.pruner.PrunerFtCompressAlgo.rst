mindspore_gs.pruner.PrunerFtCompressAlgo
========================================

.. py:class:: mindspore_gs.pruner.PrunerFtCompressAlgo(config)

    `PrunerFtCompressAlgo` 是CompAlgo的子类，实现了删除冗余卷积核并对网络进行完整训练的能力。

    .. note::
        - 针对入参 `config` ，目前 `PrunerFtCompressAlgo` 是没有可选的配置项，但为了兼容性，`config` 被保留，在初始化时以空字典代替。如 `kf_pruning = PrunerFtCompressAlgo({})` 。

    参数：
        - **config** (dict) - 算法配置参数。


    .. py:method:: apply(net)

        将Konckoff网络变为剪枝后的网络。

        参数：
            - **net** (Cell) - Konckoff网络。

        返回：
            返回剪枝后的网络。

        异常：
            - **TypeError** - `net` 不是Cell。
