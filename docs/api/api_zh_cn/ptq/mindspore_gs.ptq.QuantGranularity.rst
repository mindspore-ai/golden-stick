mindspore_gs.ptq.QuantGranularity
============================================================

.. py:class:: mindspore_gs.ptq.QuantGranularity

    用于配置 MindSpore Golden Stick 后量化算法阶段的量化粒度。

    - ``PER_TENSOR`` : 配置量化粒度为 per_tensor 。
    - ``PER_CHANNEL`` : 配置量化粒度为 per_channel 。
    - ``PER_TOKEN`` : 配置量化粒度为 per_token 。
    - ``PER_GROUP`` : 配置量化粒度为 per_group 。

    .. py:method:: from_str(name: str)
        :classmethod:

        将 `name` 转换成量化粒度类型。

        参数：
            - **name** (str) - 量化粒度的字符串名。
