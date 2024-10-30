mindspore_gs.ptq.PTQConfig
============================================================

.. py:class:: mindspore_gs.ptq.PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, opname_blacklist=[], algo_args={}, weight_quant_dtype=Int8, kvcache_quant_dtype=None, act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.NONE, act_quant_granularity=QuantGranularity.PER_TENSOR, kvcache_quant_granularity=QuantGranularity.PER_CHANNEL)

    用于配置 MindSpore Golden Stick 后量化算法的数据类。

    参数：
        - **mode** (:class:`mindspore_gs.ptq.PTQMode`) - 用于配置 PTQ 算法的模式， ``QUANTIZE`` 表示为量化阶段， ``DEPLOY`` 表示为部署阶段。
        - **backend** (:class:`mindspore_gs.common.BackendTarget`) - 用于配置 PTQ 算法转换为真实量化网络后端， ``NONE`` 表示无具体后端，为通用网络。 ``ASCEND`` 表示为昇腾后端，会在网络中插入昇腾相关算子。
        - **opname_blacklist** (List[str]) - 算子名称黑名单。网络中的层如果名字和黑名单中某一项模糊匹配上，则该层不会被量化。
        - **algo_args** (Union[dict, dataclass]) - 用于配置RTN、SmoothQuant、OmniQuant等算法的超参数。
        - **act_quant_dtype** (mindspore.dtype) - 用于配置激活的量化类型。mindspore.dtype.int8表示对激活进行8bit量化，None表示不进行量化。
        - **weight_quant_dtype** (mindspore.dtype) - 用于配置权重的量化类型。mindspore.dtype.int8表示对权重进行8bit量化，None表示不进行量化。
        - **kvcache_quant_dtype** (mindspore.dtype) - 用于配置kvcache的量化类型。mindspore.dtype.int8表示对kvcache进行8bit量化，None表示不进行量化。
        - **outliers_suppression** (:class:`mindspore_gs.ptq.OutliersSuppressionType`) - 用于配置离群值抑制方法。OutliersSuppressionType.SMOOTH 表示使用 类似于SmoothQuant算法中的smooth方法来抑制离群值，OutliersSuppressionType.NONE 作为默认值表示不对异常值执行任何操作。
        - **act_quant_granularity** (:class:`mindspore_gs.ptq.QuantGranularity`) - 用于配置激活的量化粒度。目前激活只支持QuantGranularity.PER_TENSOR和QuantGranularity.PER_TOKEN。
        - **kvcache_quant_granularity** (:class:`mindspore_gs.ptq.QuantGranularity`) - 用于配置kvcache的量化粒度。目前kvcache只支持QuantGranularity.PER_CHANNEL和QuantGranularity.PER_TOKEN。
        - **weight_quant_granularity** (:class:`mindspore_gs.ptq.QuantGranularity`) - 用于配置weight的量化粒度。目前weight只支持QuantGranularity.PER_CHANNEL和QuantGranularity.PER_GROUP。
        - **group_size** (int) - per_group量化时的group_size大小，建议使用64或128。
    异常：
        - **ValueError** - `mode` 输入不在 [PTQMode.QUANTIZE, PTQMode.DEPLOY] 中。
        - **ValueError** - `backend` 输入不在 [BackendTarget.NONE, BackendTarget.ASCEND] 中。
        - **TypeError** - `opname_blacklist` 不是一个字符串的列表。
        - **ValueError** - `act_quant_dtype` 输入不在 [mindspore.dtype.int8, None] 中。
        - **ValueError** - `weight_quant_dtype` 输入不在 [mindspore.dtype.int8, None] 中。
        - **ValueError** - `kvcache_quant_type` 输入不在 [mindspore.dtype.int8, None] 中。
        - **TypeError** - `outliers_suppression` 不是 OutliersSuppressionType 类型。
        - **ValueError** - `act_quant_granularity` 输入不在 [QuantGranularity.PER_TENSOR, QuantGranularity.PER_TOKEN] 中。
        - **ValueError** - `kvcache_quant_granularity` 输入不在 [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TOKEN] 中。
        - **ValueError** - `act_quant_granularity` 是QuantGranularity.PER_TOKEN, 但weight_quant_dtype != msdtype.int8或act_quant_dtype != msdtype.int8。
        - **ValueError** - `kvcache_quant_granularity` 是QuantGranularity.PER_TOKEN, 但kvcache_quant_dtype != msdtype.int8。
        - **ValueError** - `weight_quant_granularity` 输入不在[QuantGranularity.PER_CHANNEL, QuantGranularity.PER_GROUP]中。
        - **ValueError** - `weight_quant_granularity` 是QuantGranularity.PER_GROUP，但 `group_size` 不在[64, 128]中。
        - **ValueError** - `weight_quant_granularity` 不是QuantGranularity.PER_GROUP，但 `group_size` 不等于0。
        - **TypeError** - `group_size` 不是Int类型。