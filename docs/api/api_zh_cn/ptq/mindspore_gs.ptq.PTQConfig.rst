mindspore_gs.ptq.PTQConfig
============================================================

.. py:class:: mindspore_gs.ptq.PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.NONE)

    用于配置 MindSpore Golden Stick 后量化算法的数据类。

    参数：
        - **mode** (:class:`mindspore_gs.ptq.PTQMode`) - 用于配置 PTQ 算法的模式， ``QUANTIZE`` 表示为量化阶段， ``DEPLOY`` 表示为部署阶段。
        - **backend** (:class:`mindspore_gs.common.BackendTarget`) - 用于配置 PTQ 算法转换为真实量化网络后端， ``NONE`` 表示无具体后端，为通用网络。 ``ASCEND`` 表示为昇腾后端，会在网络中插入昇腾相关算子。
        - **opname_blacklist** (List[str]): 算子名称黑名单。网络中的层如果名字和黑名单中某一项模糊匹配上，则该层不会被量化。

    异常：
        - **ValueError** - `mode` 输入不在 [PTQMode.QUANTIZE, PTQMode.DEPLOY] 中。
        - **ValueError** - `backend` 输入不在 [BackendTarget.NONE, BackendTarget.ASCEND] 中。
        - **ValueError** - `opname_blacklist` 不是一个字符串的列表。
