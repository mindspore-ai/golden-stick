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

    mindspore_gs.ptq.PTQMode
    mindspore_gs.ptq.OutliersSuppressionType
    mindspore_gs.ptq.PrecisionRecovery
    mindspore_gs.ptq.PTQConfig

网络适配层
-----------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.NetworkHelper
    mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper
    mindspore_gs.ptq.network_helpers.mf_net_helpers.MFParallelLlama2Helper

训练后量化算法
-------------------

.. mscnautosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQ
    mindspore_gs.ptq.RoundToNearest
