# Copyright 2024-2025 Huawei Technologies Co., Ltd
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

import os
import copy
import gc
from types import MethodType

from mindspore import ops as msops
from mindspore import nn, Tensor, Parameter, mint
# from mindformers.modules.layers import Linear
# from mindformers.parallel_core.inference.tensor_parallel.layers import (
#     ColumnParallelLinear as McoreColumnParallelLinear, RowParallelLinear as McoreRowParallelLinear)
# from mindformers.parallel_core.inference.tensor_parallel.layers import QKVParallelLinear
# from mindformers.parallel_core.inference.tensor_parallel.layers import ReplicatedLinear
# from mindformers.parallel_core.inference.tensor_parallel.layers import MergedColumnParallelLinear
# from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
#     ColumnParallelGroupedLinear,
#     RowParallelGroupedLinear
# )
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.hal import (SmoothMatmul, SmoothMatmulForDeploy)
from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoothQuant
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.ptq.hal import ParallelType
# from .parallel_minmax import (
#     get_smooth_x_obs_min_max_op,
#     get_min_max_op)
from .linear_wrapper import WrapperLinearCell


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        parallel_type = None
        self.transpose_b = True
        self.compute_type = self.layer.weight.dtype
        self.is_rowparallel = parallel_type == ParallelType.ROW_PARALLEL
        self.is_colparallel = parallel_type == ParallelType.COL_PARALLEL

        self.x_obs_max, self.x_obs_min = msops.max, msops.min
        self.w_obs_max, self.w_obs_min = msops.max, msops.min # get_min_max_op(cfg.tp_size, self.is_colparallel)

    def _transpose_b(self):
        return True

    def _calc_smooth_scale(self, alpha, **kwargs):
        raise NotImplementedError

    def _apply_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._transpose_b():
            weight_scale = msops.transpose(weight_scale, (1, 0))
        orin_dtype = self._layer.weight.dtype
        weight = msops.mul(self._layer.weight, weight_scale)
        weight = msops.cast(weight, orin_dtype)
        msops.assign(self._layer.weight, weight)
        logger.debug(f"SmoothLinearCell: smoothed_weight of Layer({self._layer_name}) is {{{self._layer.weight.shape}, "
                     f"{self._layer.weight.dtype}}}")

    def _apply_group_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        org_shape = self._layer.weight.shape
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._transpose_b():
            weight_scale = msops.transpose(weight_scale, (1, 0))
            # [num_experts, ic, oc] -> [ic, num_experts * oc]
            weight = msops.transpose(self._layer.weight.data, (1, 0, 2)).reshape((org_shape[1], -1))
        else:
            # [num_experts, oc, ic] -> [num_experts * oc, ic]
            weight = self._layer.weight.data.reshape((-1, org_shape[-1]))

        orin_dtype = self._layer.weight.dtype
        weight = msops.mul(weight, weight_scale)
        weight = msops.cast(weight, orin_dtype)

        if not self._transpose_b():
            # [ic, num_experts * oc] -> [num_experts, ic, oc]
            weight = weight.reshape((org_shape[1], org_shape[0], org_shape[2]))
            weight = msops.transpose(weight, (1, 0, 2))
        else:
            # [num_experts * oc, ic] -> [num_experts, oc, ic]
            weight = weight.reshape(org_shape)
        self._layer.weight.set_data(weight)
        logger.debug(f"SmoothLinearCell: smoothed_group_weight of Layer({self._layer_name})"
                     f"is {{{self._layer.weight.shape}, "
                     f"{self._layer.weight.dtype}}}")

    def _apply_smooth(self, smooth_scale):
        """_apply_smooth"""

        # self._apply_act_smooth(smooth_scale)
        self._apply_weight_smooth(smooth_scale)
        self.layer.smooth_scale = Parameter(smooth_scale.astype(self.compute_type))

    def process(self):
        super(SmoothLinearCell, self).process()
        smooth_scale = self._calc_smooth_scale(self.cfg.algo_args.get('alpha', 0.5))
        logger.debug(f"SmoothLinearCell: smooth_scale of Layer({self._layer_name}) is {{{smooth_scale.shape}, "
                     f"{smooth_scale.dtype}}}")
        self._apply_smooth(smooth_scale)
        return self.layer



class SmoothQuantLinearCell(SmoothLinearCell):
    """SmoothLinearCell"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.SMOOTH
        LinearSmoothQuant.reg_layer_map(nn.Dense, SmoothQuantLinearCell, SmoothChecker())
        LinearSmoothQuant.reg_layer_map(mint.nn.Linear, SmoothQuantLinearCell, SmoothChecker())

    def _calc_smooth_scale(self, alpha, **kwargs):
        """_calc_smooth_scale"""
        shift_values = kwargs.get('shift_values', None)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|input0_alpha", Tensor(alpha))
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|input1_activation_inputs",
                                  self.cat_samples)
        act_max = msops.maximum(
            msops.abs(
                self.x_obs_max(self.cat_samples - shift_values if shift_values is not None else self.cat_samples, 0)[0]
            ),
            msops.abs(
                self.x_obs_min(self.cat_samples - shift_values if shift_values is not None else self.cat_samples, 0)[0]
            ),
        )
        logger.debug(f"SmoothLinearCell: act_max of Layer({self._layer_name}) is {{{act_max.shape}, {act_max.dtype}}}")
        input_max_pow = msops.pow(act_max, alpha)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|output0_activation_minmax_pow",
                                  input_max_pow)
        weight_smooth_minmax_axis = -2 if self._transpose_b() else -1
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input0_alpha", Tensor(alpha))
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input1_weight", self.layer.weight)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input2_weight_minmax_axis",
                                  Tensor(weight_smooth_minmax_axis))
        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.layer.weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.layer.weight, weight_smooth_minmax_axis)[0]))
        if len(weight_max.shape) == 2:
            weight_max = self.w_obs_max(weight_max, 0)[0]
        elif len(weight_max.shape) > 2:
            raise RuntimeError(f'Not support rank of weight bigger than 3, got {len(weight_max.shape)}.')
        logger.debug(f"SmoothLinearCell: weight_max of Layer({self._layer_name}) is {{{weight_max.shape}, "
                     f"{weight_max.dtype}}}")
        weight_max_pow = msops.pow(weight_max, 1 - alpha)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|output0_weight_max_pow", weight_max_pow)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|input0_input_max_pow", input_max_pow)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|input1_weight_max_pow", weight_max_pow)
        smooth_scale = msops.div(input_max_pow, weight_max_pow).clamp(1e-5)
        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[input_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|output0_smooth_scale", smooth_scale)
        return smooth_scale

    def _quant_info(self):
        return "SmoothQuant"
