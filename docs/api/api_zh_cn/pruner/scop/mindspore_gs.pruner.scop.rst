mindspore_gs.pruner.scop
========================

SCOP算法包含两个阶段：利用高仿数据来学习发现冗余卷积核的 `Knockoff Data <https://www.mindspore.cn/golden_stick/docs/zh-CN/master/pruner/scop.html#knockoff-data%E9%98%B6%E6%AE%B5>`_ 阶段以及删除冗余卷积核进行完整训练的 `Finetune <https://www.mindspore.cn/golden_stick/docs/zh-CN/master/pruner/scop.html#finetune%E9%98%B6%E6%AE%B5>`_ 阶段。Knockoff Data阶段算法如下：

.. include:: mindspore_gs.pruner.scop.PrunerKfCompressAlgo.rst

Finetune阶段算法如下：

.. include:: mindspore_gs.pruner.scop.PrunerFtCompressAlgo.rst
