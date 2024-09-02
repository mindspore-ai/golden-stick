mindspore_gs.ptq
=========================

Post training quantization algorithms.

.. code-block::

    import mindspore_gs.ptq as ptq

PTQ Config
-------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQConfig

PTQMode Enum
-------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQMode

OutliersSuppressionType Enum
------------------------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.OutliersSuppressionType

NetworkHelper
-------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.NetworkHelper
    mindspore_gs.ptq.network_helpers.mf_net_helpers.MFLlama2Helper
    mindspore_gs.ptq.network_helpers.mf_parallel_llama2_helper.MFParallelLlama2Helper

RoundToNearest Algorithm
--------------------------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.RoundToNearest

PTQ Algorithm
-------------

.. autosummary::
    :toctree: ptq
    :nosignatures:
    :template: classtemplate.rst

    mindspore_gs.ptq.PTQ

