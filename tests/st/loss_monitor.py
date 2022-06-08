# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Golden stick ST utils"""

import time
import numpy as np

from mindspore.train.callback import Callback
from mindspore import Tensor


class LossMonitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> LossMonitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None, step_threshold=10):
        super(LossMonitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
        self.step_threshold = step_threshold

    def epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        A backwards compatibility alias for `on_train_epoch_begin` and `on_eval_epoch_begin`.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.
        A backwards compatibility alias for `on_train_epoch_end` and `on_eval_epoch_end`.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:8.6f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))
        self.epoch_mseconds = epoch_mseconds

    def step_begin(self, run_context):
        """
        Called before each step beginning.
        A backwards compatibility alias for `on_train_step_begin` and `on_eval_step_begin`.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.step_time = time.time()

    def step_end(self, run_context):
        """
        Called after each step finished.
        A backwards compatibility alias for `on_train_step_end` and `on_eval_step_end`.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:8.6f}/{:8.6f}], time:[{:5.3f}], lr:[{:5.5f}]".format(
            cb_params.cur_epoch_num, cb_params.epoch_num, cur_step_in_epoch +
            1, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))

        if cb_params.cur_step_num == self.step_threshold:
            run_context.request_stop()
