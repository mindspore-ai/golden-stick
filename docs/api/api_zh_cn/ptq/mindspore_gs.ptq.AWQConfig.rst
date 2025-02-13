mindspore_gs.ptq.AWQConfig
============================================================

.. py:class:: mindspore_gs.ptq.AWQConfig(duo_scaling=True, smooth_alpha=[i/20 for i in range(20)], weight_clip_ratio=[1- i/20 for i in range(10)])

    用于配置 AWQ 量化算法的数据类。

    参数：
        - **duo_scaling** (bool，可选) - 表示是否使用激活值和权重计算 scale 。默认值： ``True``。
        - **smooth_alpha** (List[float]，可选) - 表示 smooth search 的超参数。默认值： ``[i/20 for i in range(20)]``。
        - **weight_clip_ratio** (List[float]，可选) - 表示 clip search 的超参数。默认值： ``[1-i/20 for i in range(10)]``。

    异常：
        - **TypeError** - `duo_scaling` 输入不是 bool 类型。
        - **TypeError** - `smooth_alpha` 输入不是 float 或者 list 类型。
        - **TypeError** - `weight_clip_ratio` 输入不是 float 或者 list 类型。
        - **ValueError** - `smooth_alpha` 输入的值小于0，或大于1。
        - **ValueError** - `weight_clip_ratio` 输入的值小于0，或大于1。
