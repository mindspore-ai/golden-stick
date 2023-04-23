mindspore_gs
=============

.. py:class:: mindspore_gs.CompAlgo(config=None)

    金箍棒中算法的基类。

    参数：
        - **config** (dict) - 模型压缩的用户配置，默认值为 ``None``。算法相关配置由派生类设置，基类属性列举如下：

          - **save_mindir** (bool) - 如果为 ``True``，则在训练后自动导出 MindIR，否则不导出。 默认值：``False``。
          - **save_mindir_path** (str) - 导出MindIR的路径，路径包括目录和文件名，可以是相对路径或绝对路径，用户需要保证写入权限。默认值：``'./network'``。

    .. py:method:: apply(network: Cell)

        定义如何压缩输入的 `network` 。此方法必须由所有算法子类重写。

        参数：
            - **network** (Cell) - 将被压缩的网络。

        返回：
            压缩后的网络。

    .. py:method:: callbacks(*args, **kwargs)

        定义训练时需要完成的任务。算法子类必须在子回调函数的最后调用基类回调函数。

        参数：
            - **args** (Union[list, tuple, optional]) - 配置给函数的参数。
            - **kwargs** (Union[dict, optional]) - 关键字参数。

        返回：
            回调实例的列表。

    .. py:method:: convert(net_opt: Cell, ckpt_path="")

        定义如何在导出到MindIR之前将压缩网络转换为标准网络。

        参数：
            - **net_opt** (Cell) - 要转换的网络，即 `CompAlgo.apply` 的输出。
            - **ckpt_path** (str) - `net_opt` 权重文件的路径。默认值为 ``""``，表示不将权重文件加载到 `net_opt` 。

        返回：
            转换后的网络实例。

    .. py:method:: loss(loss_fn: callable)

        定义如何调整算法的损失函数。如果当前算法不关心损失函数，子类不需要复写此方法。

        参数：
            - **loss_fn** (callable) - 原损失函数。

        返回：
            调整后的损失函数。

    .. py:method:: set_save_mindir(save_mindir: bool)

        设置训练后是否自动导出MindIR。

        参数：
            - **save_mindir** (bool) - 如为真，则在训练后自动导出MindIR，否则不自动导出。

        异常：
            - **TypeError** - `save_mindir` 数据类型不是bool。

    .. py:method:: set_save_mindir_path(save_mindir_path: str)

        设置导出MindIR的路径，仅当 `save_mindir` 为True时才生效。

        参数：
            - **save_mindir_path** (str) - 导出MindIR的路径，路径包括目录和文件名，可以是相对路径，也可以是绝对路径，用户需要保证写入权限。

        异常：
            - **ValueError** - `save_mindir_path` 不是非空字符串。
