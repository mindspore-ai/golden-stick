mindspore_gs.ptq.PrecisionRecovery
============================================================

.. py:class:: mindspore_gs.ptq.PrecisionRecovery

    PTQ量化算法的精度补偿枚举类。

    - ``GPTQ`` : 使用GPTQ算法做精度补偿。
    - ``NONE`` : 不做精度补偿。

    .. py:method:: from_str(name: str)
        :classmethod:

        将 `name` 转换成精度补偿算法类型。
