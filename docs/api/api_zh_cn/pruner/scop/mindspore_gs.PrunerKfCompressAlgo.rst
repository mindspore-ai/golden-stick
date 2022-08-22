mindspore_gs.PrunerKfCompressAlgo
=================================

.. py:class:: mindspore_gs.PrunerKfCompressAlgo(config)

    `PrunerKfCompressAlgo` 是GoldenStick的派生类，继承于基类 `CompAlgo` ，用于生成Knockoff阶段的网络。

    .. note::
        - 针对入参 `config` ，目前 `PrunerKfCompressAlgo` 是没有可选的配置项，但为了兼容性， `config` 被保留，在初始化时以空字典代替。如 `kf_pruning = PrunerKfCompressAlgo({})` 。

    参数：
        - **config** (dict) - 算法配置参数。


    .. py:method:: apply(net)

        将网络变成Konckoff网络。

        参数：
            - **net** (Cell) - 原始待剪枝网络。

        返回：
            返回变换后的Konckoff网络。
        
        异常：
            - **TypeError** - `net` 不是Cell。
