mindspore_gs.ptq
=========================

训练后量化算法集。

.. code-block::

    import mindspore_gs.ptq as ptq

PTQ 配置
-------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQConfig

PTQ 模式枚举
-------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQMode

异常值抑制类型枚举
--------------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.OutliersSuppressionType

精度补偿类型
--------------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PrecisionRecovery

网络适配层
-----------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.NetworkHelper
    mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper
    mindspore_gs.ptq.network_helpers.mf_net_helpers.MFParallelLlama2Helper

PTQ 算法
-------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQ

RoundToNearest 算法
--------------------------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.RoundToNearest
