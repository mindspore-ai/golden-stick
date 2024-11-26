# Copyright 2024 Huawei Technologies Co., Ltd
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
"""ptq wrapper cells for mindformers."""

from types import MethodType

from mindspore import Tensor
from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.ptq.hal import SmoothMatmul, SmoothMatmulForDeploy
from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoother
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .parallel_minmax import get_smooth_x_obs_min_max_op, get_smooth_w_obs_min_max_op
from .linear_wrapper import WrapperLinearCell


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""

    @staticmethod
    def reg_self():
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.SMOOTH

        LinearSmoother.reg_layer_map(Linear, SmoothLinearCell, SmoothChecker())
        LinearSmoother.reg_layer_map(ColumnParallelLinear, SmoothLinearCell, SmoothChecker())
        LinearSmoother.reg_layer_map(RowParallelLinear, SmoothLinearCell, SmoothChecker())

    def __init__(self, linear_name, linear, cfg, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        self.is_rowparallel = isinstance(self.layer, RowParallelLinear)
        self.is_colparallel = isinstance(self.layer, ColumnParallelLinear)
        self.is_linear = isinstance(self.layer, Linear)
        if not self.is_rowparallel and not self.is_colparallel and not self.is_linear:
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")

        self.compute_type = self.layer.dtype if self.is_linear else self.layer.compute_dtype

        self.x_obs_max, self.x_obs_min = get_smooth_x_obs_min_max_op()
        self.w_obs_max, self.w_obs_min = get_smooth_w_obs_min_max_op(cfg.tp_size, self.is_colparallel)

    def _calc_smooth_scale(self, alpha):
        """_calc_smooth_scale"""
        act_max = msops.maximum(msops.abs(self.x_obs_max(self.cat_samples, 0)[0]),
                                msops.abs(self.x_obs_min(self.cat_samples, 0)[0]))
        logger.debug(f"SmoothLinearCell: act_max of Layer({self._layer_name}) is {{{act_max.shape}, {act_max.dtype}, "
                     f"{act_max.asnumpy()}}}")
        input_max_pow = msops.pow(act_max, alpha)
        weight_smooth_minmax_axis = -2 if self.layer.transpose_b else -1
        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.layer.weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.layer.weight, weight_smooth_minmax_axis)[0]))
        logger.debug(f"SmoothLinearCell: weight_max of Layer({self._layer_name}) is {{{weight_max.shape}, "
                     f"{act_max.dtype}, {weight_max.asnumpy()}}}")
        weight_max_pow = msops.pow(weight_max, 1 - alpha)
        smooth_scale = msops.div(input_max_pow, weight_max_pow).clamp(1e-5)
        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[input_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0
        return smooth_scale

    def _apply_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._layer.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._layer.weight.dtype
        weight = msops.mul(self._layer.weight, weight_scale)
        weight = self._layer.cast(weight, orin_dtype)
        msops.assign(self._layer.weight, weight)
        logger.debug(f"SmoothLinearCell: smoothed_weight of Layer({self._layer_name}) is {{{self._layer.weight.shape}, "
                     f"{self._layer.weight.dtype}, {self._layer.weight.asnumpy()}}}")

    def _apply_act_smooth(self, smooth_scale: Tensor):
        """_apply_act_smooth"""
        self._layer.matmul = SmoothMatmul.create(self._layer_name, self._layer.matmul, smooth_scale=smooth_scale)

    def _apply_smooth(self, smooth_scale):
        """_apply_smooth"""
        self._apply_act_smooth(smooth_scale)
        self._apply_weight_smooth(smooth_scale)

    def process(self):
        super(SmoothLinearCell, self).process()
        smooth_scale = self._calc_smooth_scale(self.cfg.algo_args.get('alpha', 0.5))
        logger.debug(f"SmoothLinearCell: smooth_scale of Layer({self._layer_name}) is {{{smooth_scale.shape}, "
                     f"{smooth_scale.dtype}, {smooth_scale.asnumpy()}}}")
        self._apply_smooth(smooth_scale)

    def _apply_act_smooth_for_deploy(self, ic, compute_dtype):
        """_apply_act_smooth_by_insert_op_for_deploy"""
        self._layer.matmul = SmoothMatmulForDeploy.create(self._layer_name, self._layer.matmul, ic=ic,
                                                          compute_dtype=compute_dtype)

    def deploy(self):
        """deploy"""
        if self.cfg.mode == PTQMode.QUANTIZE or self.cfg.outliers_suppression == OutliersSuppressionType.NONE:
            return self.layer
        logger.info("insert ops for smooth quant.")
        ic = self._layer.weight.shape[1] if self._layer.transpose_b else self._layer.weight.shape[1]
        self._apply_act_smooth_for_deploy(ic, self.compute_type)
        if self.is_colparallel:
            self.layer.sharded_state_dict = MethodType(SmoothLinearCell.col_sharded_state_dict, self.layer)
        if self.is_rowparallel:
            self.layer.sharded_state_dict = MethodType(SmoothLinearCell.row_sharded_state_dict, self.layer)
        return self.layer

    @staticmethod
    # pylint: disable=W0211
    def col_sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)
        smooth_scale_shard = (1,)
        state_dict = {self.weight.name: {'shape': self.weight.shape, 'shard': w_shard},
                      self.matmul.smooth_scale.name: {'shape': self.matmul.smooth_scale.shape,
                                                      'shard': smooth_scale_shard}}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape, 'shard': (self.tensor_parallel_group_size,)}
        return state_dict

    @staticmethod
    #pylint: disable=W0211
    def row_sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1, self.tensor_parallel_group_size) if self.transpose_b else (self.tensor_parallel_group_size, 1)
        smooth_scale_shard = (self.tensor_parallel_group_size,)
        state_dict = {self.weight.name: {'shape': self.weight.shape, 'shard': w_shard},
                      self.matmul.smooth_scale.name: {'shape': self.matmul.smooth_scale.shape,
                                                      'shard': smooth_scale_shard}}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict
