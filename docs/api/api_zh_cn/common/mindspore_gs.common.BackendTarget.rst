mindspore_gs.common.BackendTarget
============================================================

.. py:class:: mindspore_gs.common.BackendTarget

    用于配置 MindSpore Golden Stick 后量化算法后端类型的枚举类。

    参数：
    - ``NONE`` : 表示无具体后端，量化网络使用 MindSpore 通用算子。
    - ``ASCEND`` : 表示昇腾后端，量化网络使用昇腾后端算子。
