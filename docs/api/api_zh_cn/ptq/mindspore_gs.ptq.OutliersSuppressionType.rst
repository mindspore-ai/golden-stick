mindspore_gs.ptq.OutliersSuppressionType
============================================================

.. py:class:: mindspore_gs.ptq.OutliersSuppressionType

    PTQ量化算法的异常值抑制枚举类。

    - ``SMOOTH`` : 使用SMOOTH方法做异常值抑制。
    - ``AWQ`` : 使用AWQ方法做异常值抑制。
    - ``OUTLIER_SUPPRESSION_PLUS`` : 使用OUTLIER_SUPPRESSION_PLUS方法做异常值抑制。
    - ``OUTLIER_SUPPRESSION_LITE`` : 使用OUTLIER_SUPPRESSION_LITE方法做异常值抑制。
    - ``NONE`` : 不做异常值抑制。

    .. py:method:: from_str(name: str)
        :classmethod:

        将 `name` 转换成异常值抑制算法类型。

        参数：
            - **name** (str) - 异常值抑制算法的字符串名。
