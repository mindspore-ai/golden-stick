mindspore_gs.ptq.GPTQQuantConfig
============================================================

.. py:class:: mindspore_gs.ptq.GPTQQuantConfig(block_size=128, desc_act=False, damp_percent=0.01, static_groups=False)

    用于配置 GPTQ 量化算法的数据类。

    参数：
        - **block_size** (int，可选) - 表示 GPTQ 算法在补偿时分块的大小。默认值： ``128``。
        - **desc_act** (bool，可选) - 表示是否对海森矩阵进行重要性排序。默认值： ``False``。
        - **damp_percent** (float，可选) - 表示在数值稳定计算时，海森矩阵对角线元素平均值的百分比。默认值： ``0.01``。
        - **static_groups** (bool，可选) - 表示是否在精度补偿之前进行 per_group 计算。默认值： ``False``。

    异常：
        - **TypeError** - `block_size` 输入不是 int 类型。
        - **TypeError** - `desc_act` 输入不是 bool 类型。
        - **TypeError** - `damp_percent` 输入不是 float 类型。
        - **TypeError** - `static_groups` 输入不是 bool 类型。
        - **ValueError** - `block_size` 输入的值小于0。
        - **ValueError** - `damp_percent` 输入的值小于0，或者大于1。
