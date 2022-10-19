mindspore_gs.quantization.SimulatedQuantizationAwareTraining
============================================================

.. py:class:: mindspore_gs.quantization.SimulatedQuantizationAwareTraining(config=None)

    模拟量化感知训练的基本实现，该算法在训练时使用伪量化节点来模拟量化计算的损失，并通过反向传播更新网络参数，使得网络参数更好地适应量化带来的损失。更多详细信息见 `神经网络量化白皮书 <https://arxiv.org/pdf/2106.08295.pdf>`_。

    参数：
        - **config** (dict) - 存储用于量化感知训练的属性，键是属性名称，值是属性值。下面列出了支持的属性：

          - **quant_delay** (Union[int, list, tuple]) - 在训练和评估期间权重和激活量化后的步数。第一个元素表示激活，第二个元素表示权重。默认值：(0, 0)。
          - **quant_dtype** (Union[QuantDtype, list, tuple]) - 用于量化权重和激活的数据类型。第一个元素表示激活，第二个元素表示权重。在实际量化推理场景中需要考虑硬件器件的精度支持。默认值：(QuantDtype.INT8, QuantDtype.INT8)。
          - **per_channel** (Union[bool, list, tuple]) - 基于层或通道的量化粒度。如果为True，则基于每个通道，否则基于每个层。第一个元素表示激活，第二个元素表示权重，第一个元素现在必须为 False。默认值：(False, Fasle)。
          - **symmetric** (Union[bool, list, tuple]) - 量化算法是否对称。如果为True，则基于对称，否则基于不对称。第一个元素表示激活，第二个元素表示权重。默认值：(False, Fasle)。
          - **narrow_range** (Union[bool, list, tuple]) - 量化算法是否使用窄范围。第一个元素表示激活，第二个元素表示权重。默认值：(False, Fasle)。
          - **enable_fusion** (bool) - 在应用量化之前是否应用融合。默认值：False。
          - **freeze_bn** (int) - `BatchNorm OP` 参数固定为全局均值和方差之后的步数。默认值：10000000。
          - **bn_fold** (bool) - 是否使用 `bn fold` 算子进行模拟推理操作。默认值：False。
          - **one_conv_fold** (bool) - 是否使用 `one conv bn fold` 算子进行模拟推理操作。默认值：True。

    异常：
        - **TypeError** - `bn_fold` ， `one_conv_fold` 或者 `enable_fusion` 的元素类型不是bool。
        - **TypeError** - `freeze_bn` 的数据类型不是int。
        - **TypeError** - `quant_delay` 的数据类型不是int，或者 `quant_delay` 存在不是int的元素。
        - **TypeError** - `quant_dtype` 的数据类型不是 `QuantDtype` ，或者 `quant_dtype` 存在不是 `QuantDtype` 的元素。
        - **TypeError** - `per_channel` 的数据类型不是bool，或者 `per_channel` 存在不是bool的元素。
        - **TypeError** - `symmetric` 的数据类型不是bool，或者 `symmetric` 存在不是bool的元素。
        - **TypeError** - `narrow_range` 的数据类型不是bool，或者 `narrow_range` 存在不是bool的元素。
        - **ValueError** - `freeze_bn` 小于0。
        - **ValueError** - `quant_delay` ， `quant_dtype` ， `per_channel` ， `symmetric` 或者 `narrow_range` 的长度大于2。
        - **ValueError** - `quant_delay` 小于0，或者 `quant_delay` 存在小于0的元素。
        - **ValueError** - `quant_dtype` 的数据类型不是 `QuantDtype.INT8` 或者 `quant_dtype` 存在不是 `QuantDtype.INT8` 的元素。
        - **ValueError** - `per_channel` 为True， 或者 `per_channel` 的第一个元素为True。

    .. py:method:: apply(network: Cell)

        按照以下步骤在 `network` 中应用SimQAT算法，使 `network` 可用于量化感知训练：

        1. 使用由网络策略定义的模式引擎融合 `network` 中的某些单元。默认融合模式：Conv2d + BatchNorm2d + ReLU， Conv2d + ReLU， Dense + BatchNorm2d + ReLU， Dense + BatchNorm2d， Dense + ReLU。

        2. 在网络中传播NetPolicy中定义的LayerPolices。

        3. 减少冗余的假量化器，即一个张量上存在两个或多个假量化器。

        4. 应用LayerPolicies将普通Cell转换为 `QuantizeWrapperCell` 。在此步骤中，我们将在网络中插入真正的假量化器。

        参数：
            - **network** (Cell) - 待量化的网络。

        返回：
            量化后的网络。

    .. py:method:: set_act_narrow_range(act_narrow_range)

        设置量化感知训练参数 `config` 的act_narrow_range值。

        参数：
            - **act_narrow_range** (bool) - 量化算法是否使用 `act_narrow_range` 。如果为True，则基于narrow_range，否则不基于narrow_range。

        异常：
            - **TypeError** - `act_narrow_range` 数据类型不是bool。

    .. py:method:: set_act_per_channel(act_per_channel)

        设置量化感知训练参数 `config` 的act_per_channel值。

        参数：
            - **act_per_channel** (bool) - 量化算法基于层还是通道。如果为True，则基于通道，否则基于层。当前只支持False。

        异常：
            - **TypeError** - `act_per_channel` 数据类型不是bool。
            - **ValueError** - `act_per_channel` 不是False。

    .. py:method:: set_act_quant_delay(act_quant_delay)

        设置量化感知训练参数 `config` 的act_quant_delay值。

        参数：
            - **act_quant_delay** (int) - 在训练和评估期间激活量化后的步数。

        异常：
            - **TypeError** - `act_quant_delay` 数据类型不是int。
            - **ValueError** - `act_quant_delay` 小于0。

    .. py:method:: set_act_quant_dtype(act_quant_dtype)

        设置量化感知训练参数 `config` 的act_quant_dtype值。

        参数：
            - **act_quant_dtype** (QuantDtype) - 激活量化的数据类型。

        异常：
            - **TypeError** - `act_quant_dtype` 数据类型不是QuantDtype。
            - **ValueError** - `act_quant_dtype` 不是 `QuantDtype.INT8` 。

    .. py:method:: set_act_symmetric(act_symmetric)

        设置量化感知训练参数 `config` 的act_symmetric值。

        参数：
            - **act_symmetric** (bool) - 量化算法是否使用激活对称。如果为True，则基于对称，否则基于不对称。

        异常：
            - **TypeError** - `act_symmetric` 数据类型不是bool。

    .. py:method:: set_bn_fold(bn_fold)

        设置量化感知训练参数 `config` 的bn_fold值。

        参数：
            - **bn_fold** (bool) - 量化算法是否使用 `bn_fold` 。

        异常：
            - **TypeError** - `bn_fold` 数据类型不是bool。

    .. py:method:: set_enable_fusion(enable_fusion)

        设置量化感知训练参数 `config` 的enable_fusion值。

        参数：
            - **enable_fusion** (bool) - 是否在量化之前进行融合，默认值为 False。

        异常：
            - **TypeError** - `enable_fusion` 数据类型不是bool。

    .. py:method:: set_freeze_bn(freeze_bn)

        设置量化感知训练参数 `config` 的freeze_bn值。

        参数：
            - **freeze_bn** (int) - `BatchNorm OP` 参数固定为全局均值和方差之后的步数。

        异常：
            - **TypeError** - `freeze_bn` 数据类型不是int。
            - **ValueError** - `freeze_bn` 小于0。

    .. py:method:: set_one_conv_fold(one_conv_fold)

        设置量化感知训练参数 `config` 的one_conv_fold值。

        参数：
            - **one_conv_fold** (bool) - 量化算法是否使用 `one_conv_fold` 。

        异常：
            - **TypeError** - `one_conv_fold` 数据类型不是bool。

    .. py:method:: set_weight_narrow_range(weight_narrow_range)

        设置量化感知训练参数 `config` 的weight_narrow_range值。

        参数：
            - **weight_narrow_range** (bool) - 量化算法是否使用权重narrow_range。如果为True，则基于narrow_range，否则不基于narrow_range。

        异常：
            - **TypeError** - `weight_narrow_range` 数据类型不是bool。

    .. py:method:: set_weight_quant_delay(weight_quant_delay)

        设置量化感知训练参数 `config` 的weight_quant_delay值。

        参数：
            - **weight_quant_delay** (int) - 在训练和评估期间权重量化后的步数。

        异常：
            - **TypeError** - `weight_quant_delay` 数据类型不是int。
            - **ValueError** - `weight_quant_delay` 小于0。

    .. py:method:: set_weight_quant_dtype(weight_quant_dtype)

        设置量化感知训练参数 `config` 的weight_quant_dtype值。

        参数：
            - **weight_quant_dtype** (QuantDtype) - 权重量化数据类型。

        异常：
            - **TypeError** - `weight_quant_dtype` 数据类型不是QuantDtype。
            - **ValueError** - `weight_quant_dtype` 不是 `QuantDtype.INT8` 。

    .. py:method:: set_weight_symmetric(weight_symmetric)

        设置量化感知训练参数 `config` 的weight_symmetric值。

        参数：
            - **weight_symmetric** (bool) - 量化算法是否使用权重对称。如果为True，则基于对称，否则基于不对称。

        异常：
            - **TypeError** - `weight_symmetric` 数据类型不是bool。

   .. py:method:: convert(net_opt, ckpt_path＝"")

        将量化网络｀net_opt｀转换为标准网络，后续导出成MindIR用于部署。

        参数：
            - **net_opt** (Cell) - 经过量化算法apply之后的网络。
            - **ckpt_path** (str) - 网络的checkpoint file文件路径，默认值为空，表示不加载。

        异常：
            - **TypeError** - `net_opt` 数据类型不是Cell。
            - **TypeError** - `ckpt_path` 数据类型不是str。
            - **ValueError** - `ckpt_path` 非空但不是有效路径。

        返回：
            转换后的网络。