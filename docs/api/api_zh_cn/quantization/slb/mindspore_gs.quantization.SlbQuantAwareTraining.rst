mindspore_gs.quantization.SlbQuantAwareTraining
===============================================

.. py:class:: mindspore_gs.quantization.SlbQuantAwareTraining(config=None)

    SLB(Searching for Low-Bit Weights in Quantized Neural Networks)算法的实现，该算法将量化神经网络中的离散权重作为可搜索的变量，并实现了一种微分方法去精确的实现该搜索。具体来说，是将每个权重表示为在离散值集上的概率分布，通过训练来优化该概率分布，最终具有最高概率的离散值就是搜索的结果，也就是量化的结果。更多详细信息见 `Searching for Low-Bit Weights in Quantized Neural Networks <https://arxiv.org/pdf/2009.08695.pdf>`_。

    .. note::
        - 本方法会调用其它接口来设置参数，所以报错时需要参考其他的接口，比如 `quant_dtype` 要参考 `set_weight_quant_dtype` 和 `set_act_quant_dtype`。

    参数：
        - **config** (dict) - 以字典的形式存放用于量化训练的属性，默认值为 ``None``。下面列出了受支持的属性：

          - **quant_dtype** (Union[QuantDtype, list(QuantDtype), tuple(QuantDtype)]) - 用于量化权重和激活的数据类型。类型为 `QuantDtype` 或包含两个 `QuantDtype` 的list或者tuple。如果 `quant_dtype` 是一个 `QuantDtype` ，则会被复制成包含两个 `QuantDtype` 的list。第一个元素表示激活的量化数据类型，第二个元素表示权重的量化数据类型。在实际量化推理场景中需要考虑硬件器件的精度支持。当前权重量化支持1、2、4比特，激活量化支持8比特。默认值：``(QuantDtype.INT8, QuantDtype.INT1)``。
          - **enable_act_quant** (bool) - 在训练中是否开启激活量化。默认值：``False``。
          - **enable_bn_calibration** (bool) - 在训练中是否开启BN层矫正功能。默认值：``False``。
          - **epoch_size** (int) - 训练的总epoch数。
          - **has_trained_epoch** (int) - 预训练的epoch数。
          - **t_start_val** (float) - 温度初始值。默认值：``1.0``。
          - **t_start_time** (float) - 温度开始变化时间。默认值：``0.2``。
          - **t_end_time** (float) - 温度停止变化时间。默认值：``0.6``。
          - **t_factor** (float) - 温度变化因子。默认值：``1.2``。

    异常：
        - **TypeError** - `quant_dtype` 的数据类型不是 `QuantDtype` ，或者 `quant_dtype` 存在不是 `QuantDtype` 的元素。
        - **TypeError** - `enable_act_quant` 或者 `enable_bn_calibration` 的数据类型不是bool。
        - **ValueError** - `quant_dtype` 的长度大于2。
        - **TypeError** - `epoch_size` 或 `has_trained_epoch` 的数据类型不是int。
        - **TypeError** - `t_start_val` 、 `t_start_time`、 `t_end_time` 或 `t_factor` 的数据类型不是float。
        - **ValueError** - `epoch_size` 小于等于0。
        - **ValueError** - `has_trained_epoch` 小于0。
        - **ValueError** - `t_start_val` 或 `t_factor` 小于等于0.0。
        - **ValueError** - `t_start_time` 或 `t_end_time` 小于0.0。
        - **ValueError** - `t_start_time` 或 `t_end_time` 大于1.0。       

    .. py:method:: apply(network: Cell, **kwargs)

        按照下面4个步骤对给定网络应用量化算法，得到带有伪量化节点的网络。

        1. 使用网络策略中定义的模式引擎在给定网络中融合特定的单元。
        2. 传播通过单元定义的层策略。
        3. 当量化器冗余时，减少冗余的伪量化器。
        4. 应用层策略将正常 `Cell` 转换为 `QuantizeWrapperCell` 。

        参数：
            - **network** (Cell) - 即将被量化的网络。
            - **kwargs** (Dict) - 用于子类的可扩展入参。

        返回：
            在原网络定义的基础上，修改需要量化的网络层后生成带有伪量化节点的网络。

    .. py:method:: callbacks(model: Model, dataset: Dataset)

        定义SLB量化算法特有的一些callbacks，其中包括用于调节温度因子的callback。

        参数：
            - **model** (Model) - 经过算法修改后的网络构造的mindspore的Model对象。
            - **dataset** (Dataset) - 加载了特定数据集的Dataset对象。

        返回：
            SLB量化算法特有的一些callbacks的列表。

        异常：
            - **RuntimeError** - `epoch_size` 没有初始化。
            - **RuntimeError** - `has_trained_epoch` 没有初始化。
            - **ValueError** - `epoch_size` 小于等于 `has_trained_epoch` 。
            - **ValueError** - `t_end_time` 小于 `t_start_time` 。
            - **TypeError** - `model` 的数据类型不是 `mindspore.train.Model`。
            - **TypeError** - `dataset` 的数据类型不是 `mindspore.dataset.Dataset`。

    .. py:method:: convert(net_opt: Cell, ckpt_path="")

        定义将SLB量化网络转换成适配MindIR的标准网络的具体实现。

        参数：
            - **net_opt** (Cell) - 经过SLB量化算法量化后的网络。
            - **ckpt_path** (str) - checkpoint文件的存储路径，为空时不加载，默认值为 ``""``。

        返回：
            能适配MindIR的标准网络。

        异常：
            - **TypeError** - `net_opt` 的数据类型不是 `mindspore.nn.Cell`。
            - **TypeError** - `ckpt_path` 的数据类型不是str。
            - **ValueError** - `ckpt_path` 不为空，但不是有效文件。
            - **RuntimeError** - `ckpt_path` 是有效文件，但加载失败。

    .. py:method:: set_act_quant_dtype(act_quant_dtype=QuantDtype.INT8)

        设置激活量化的数据类型。

        参数：
            - **act_quant_dtype** (QuantDtype) - 激活量化的数据类型。默认值：``QuantDtype.INT8``。

        异常：
            - **TypeError** - `act_quant_dtype` 的数据类型不是QuantDtype。
            - **ValueError** - `act_quant_dtype` 不是 `QuantDtype.INT8` 。

    .. py:method:: set_enable_act_quant(enable_act_quant=False)

        设置是否开启激活量化。

        参数：
            - **enable_act_quant** (bool) - 在训练中是否开启激活量化。默认值：``False``。

        异常：
            - **TypeError** - `enable_act_quant` 的数据类型不是bool。

    .. py:method:: set_enable_bn_calibration(enable_bn_calibration=False)

        设置是否开启BatchNorm层矫正功能。

        参数：
            - **enable_bn_calibration** (bool) - 在训练中是否开启BatchNorm层矫正功能。默认值：``False``。

        异常：
            - **TypeError** - `enable_bn_calibration` 的数据类型不是bool。

    .. py:method:: set_epoch_size(epoch_size)

        设置训练的总epoch数。

        参数：
            - **epoch_size** (int) - 训练的总epoch数。

        异常：
            - **TypeError** - `epoch_size` 的数据类型不是int。
            - **ValueError** - `epoch_size` 小于等于0。

    .. py:method:: set_has_trained_epoch(has_trained_epoch)

        设置预训练的epoch数。

        参数：
            - **has_trained_epoch** (int) - 预训练的epoch数。

        异常：
            - **TypeError** - `has_trained_epoch` 的数据类型不是int。
            - **ValueError** - `has_trained_epoch` 小于0。

    .. py:method:: set_t_end_time(t_end_time=0.6)

        设置温度停止变化时间。

        参数：
            - **t_end_time** (float) - 温度停止变化时间。默认值：``0.6``。

        异常：
            - **TypeError** - `t_end_time` 的数据类型不是float。
            - **ValueError** - `t_end_time` 小于0.0或大于1.0。

    .. py:method:: set_t_factor(t_factor=1.2)

        设置温度变化因子。

        参数：
            - **t_factor** (float) - 温度变化因子。默认值：``1.2``。

        异常：
            - **TypeError** - `t_factor` 的数据类型不是float。
            - **ValueError** - `t_factor` 小于等于0.0。

    .. py:method:: set_t_start_time(t_start_time=0.2)

        设置温度开始变化时间。

        参数：
            - **t_start_time** (float) - 温度开始变化时间。默认值：``0.2``。

        异常：
            - **TypeError** - `t_start_time` 的数据类型不是float。
            - **ValueError** - `t_start_time` 小于0.0或大于1.0。 

    .. py:method:: set_t_start_val(t_start_val=1.0)

        设置温度初始值。

        参数：
            - **t_start_val** (float) - 温度初始值。默认值：``1.0``。

        异常：
            - **TypeError** - `t_start_val` 的数据类型不是float。
            - **ValueError** - `t_start_val` 小于等于0.0。               

    .. py:method:: set_weight_quant_dtype(weight_quant_dtype=QuantDtype.INT1)

        设置权重量化的数据类型。

        参数：
            - **weight_quant_dtype** (QuantDtype) - 权重量化的数据类型。默认值：``QuantDtype.INT1``。

        异常：
            - **TypeError** - `weight_quant_dtype` 的数据类型不是QuantDtype。
            - **ValueError** - `weight_quant_dtype` 不是 `QuantDtype.INT1` 、 `QuantDtype.INT2` 和 `QuantDtype.INT4` 中的一种。 
