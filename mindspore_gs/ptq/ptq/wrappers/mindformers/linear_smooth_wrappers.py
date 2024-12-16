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

import copy
import enum
from types import MethodType
from typing import Optional

from mindspore import Tensor, nn
from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import (
    InnerPTQConfig, PTQMode,
    OutliersSuppressionType,
    QuantGranularity)
from mindspore_gs.ptq.ptq.hal import SmoothMatmul, SmoothMatmulForDeploy
from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoothQuant, LinearAutoSmoother
from mindspore_gs.ptq.ptq.wrapper_cell import Checker, SearchInputs
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from .parallel_minmax import (
    get_smooth_x_obs_min_max_op,
    get_w_sum_op,
    get_min_max_op)
from .linear_wrapper import WrapperLinearCell


class SmoothMethod(enum.Enum):
    NONE = 0
    SMOOTH_QUANT = 1
    AWQ = 2
    AUTO = 3


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""

    def __init__(self, linear_name, linear, cfg, network_helper, **kwargs):
        super().__init__(linear_name, linear, cfg, network_helper, **kwargs)
        self.is_rowparallel = isinstance(self.layer, RowParallelLinear)
        self.is_colparallel = isinstance(self.layer, ColumnParallelLinear)
        self.is_linear = isinstance(self.layer, Linear)
        if not self.is_rowparallel and not self.is_colparallel and not self.is_linear:
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")

        self.compute_type = self.layer.dtype if self.is_linear else self.layer.compute_dtype

        self.x_obs_max, self.x_obs_min = get_smooth_x_obs_min_max_op()
        self.w_obs_max, self.w_obs_min = get_min_max_op(cfg.tp_size, self.is_colparallel)
        self.smooth_method = self._get_smooth_method()

    def _get_smooth_method(self):
        raise NotImplementedError

    def _calc_smooth_scale(self, alpha):
        if self.smooth_method is SmoothMethod.SMOOTH_QUANT:
            return self._calc_smooth_quant_smooth_scale(alpha)
        if self.smooth_method is SmoothMethod.AWQ:
            return self._calc_awq_smooth_scale(alpha)
        raise RuntimeError(f"Unsupported SmoothMethod: {self.smooth_method}")

    def _calc_smooth_quant_smooth_scale(self, alpha):
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

    def _calc_awq_smooth_scale(self, alpha):
        """_calc_smooth_scale"""
        if self.cfg.algo_args.get("duo_scaling", True):
            x_pow = msops.pow(self.x_mean, alpha)
            w_pow = msops.pow(self.w_mean, 1 - alpha) + 1e-4
            smooth_scale = (x_pow / w_pow).clamp(min=1e-4)
        else:
            smooth_scale = msops.pow(self.x_mean, alpha).clamp(1e-4).reshape(-1)

        minmax_norm = msops.sqrt(self.scale_max(smooth_scale)[0] * self.scale_min(smooth_scale)[0])
        smooth_scale = smooth_scale / minmax_norm
        smooth_scale[self.x_mean == 0] = 1
        smooth_scale[self.w_mean == 0] = 1
        logger.debug(f"AWQSmoothLinearCell: search scale alpha {alpha}, smooth scale of Layer({self._layer_name}) "
                     f"is {{{smooth_scale.shape}, {smooth_scale.dtype}, {smooth_scale.asnumpy()}}}")
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


class SmoothQuantLinearCell(SmoothLinearCell):
    """SmoothLinearCell"""
    @staticmethod
    def reg_self():
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.SMOOTH

        LinearSmoothQuant.reg_layer_map(Linear, SmoothQuantLinearCell, SmoothChecker())
        LinearSmoothQuant.reg_layer_map(ColumnParallelLinear, SmoothQuantLinearCell, SmoothChecker())
        LinearSmoothQuant.reg_layer_map(RowParallelLinear, SmoothQuantLinearCell, SmoothChecker())

    def _get_smooth_method(self):
        return SmoothMethod.SMOOTH_QUANT


class AWQLinearCell(SmoothLinearCell):
    """SmoothLinearCell"""
    @staticmethod
    def reg_self():
        class AWQChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return False

        LinearSmoothQuant.reg_layer_map(Linear, AWQLinearCell, AWQChecker())
        LinearSmoothQuant.reg_layer_map(ColumnParallelLinear, AWQLinearCell, AWQChecker())
        LinearSmoothQuant.reg_layer_map(RowParallelLinear, AWQLinearCell, AWQChecker())

    def _get_smooth_method(self):
        return SmoothMethod.AWQ


class SearchLinearCell(nn.Cell):
    """SearchLinearCell"""
    def __init__(self, search_inputs: Optional[SearchInputs] = None):
        super().__init__()
        self.target_layer = search_inputs.layer if search_inputs else None
        self.target_args = search_inputs.layer_args if search_inputs else None
        self.target_kwargs = search_inputs.layer_kwargs if search_inputs else None

    def _target_forward(self):
        if self.target_layer is None:
            return None
        # pylint: disable=not-callable
        return self.target_layer(*self.target_args, **self.target_kwargs)

    def _try_next(self) -> tuple:
        raise NotImplementedError

    def _loss(self, ground, pred):
        raise NotImplementedError

    def _settle_best(self, best_hyper_param: tuple):
        raise NotImplementedError

    def search_best(self):
        """search_best"""
        ground = self._target_forward()
        min_loss = float("inf")
        best_hyper_param = None
        while True:
            hyper_param = self._try_next()
            if not hyper_param:
                break
            pred = self._target_forward()
            loss = self._loss(ground, pred)
            if loss < min_loss:
                min_loss = loss
                best_hyper_param = hyper_param
        if not best_hyper_param:
            raise RuntimeError(f"No search space found.")
        self._settle_best(best_hyper_param)


class AWQSmoothLinearCell(AWQLinearCell):
    """AWQLinearCell"""

    @staticmethod
    def reg_self():
        class AWQSmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.AWQ

        LinearAutoSmoother.reg_layer_map(Linear, AWQSmoothLinearCell, AWQSmoothChecker())
        LinearAutoSmoother.reg_layer_map(ColumnParallelLinear, AWQSmoothLinearCell, AWQSmoothChecker())
        LinearAutoSmoother.reg_layer_map(RowParallelLinear, AWQSmoothLinearCell, AWQSmoothChecker())

    def __init__(self, linear_name, linear, cfg, network_helper, **kwargs):
        super().__init__(linear_name, linear, cfg, network_helper, **kwargs)

        if cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, False)
        else:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)
        self.w_sum = get_w_sum_op(cfg.tp_size, self.is_colparallel)
        self.scale_max, self.scale_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)

        rank = len(linear.weight.shape)
        self.ic_axis = rank - 1 if linear.transpose_b else rank - 2
        self.oc_axis = rank - 2 if linear.transpose_b else rank - 1
        self.oc = linear.weight.shape[self.oc_axis]
        self.fp16_weight = copy.deepcopy(self._layer.weight)

        self.decoder = kwargs.get("decoder_layer", None)
        self.forward_module = None
        self.args = kwargs.get("layer_args", None)
        self.kwargs = kwargs.get("layer_kwargs", None)

        self.w_mean = None
        self.x_mean = None

    def _get_mean_weight(self, weight, axis):
        """_get_mean_weight"""
        need_comm = self.cfg.tp_size is not None and self.cfg.tp_size > 1
        if need_comm and not self.is_linear:
            w_sum = self.w_sum(weight, axis)
            if self.is_colparallel:
                w_mean = w_sum / (self.oc * 2)
            elif self.is_rowparallel:
                w_mean = w_sum / self.oc
            else:
                raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported.")
        else:
            w_mean = msops.mean(weight, axis=axis)
        return w_mean

    def _get_statistic_data(self):
        """_get_statistic_data"""

        # compute mean of normalised weights
        org_shape = self._layer.weight.shape
        if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            # group in input channel
            dst_shape = (-1, self.cfg.group_size) if self._layer.transpose_b else (self.cfg.group_size, -1)
            weight = self._layer.weight.reshape(dst_shape)
        else:
            weight = self._layer.weight
        w_max = msops.max(msops.abs(weight), self.ic_axis, keepdims=True)[0] + 1e-6
        w_scale = msops.abs(weight) / w_max
        w_scale = w_scale.reshape(org_shape)
        self.w_mean = self._get_mean_weight(w_scale, self.oc_axis)
        logger.debug(f"AWQSmoothLinearCell: w_mean of Layer({self._layer_name}) is {{{self.w_mean.shape}, "
                     f"{self.w_mean.dtype}, {self.w_mean.asnumpy()}}}")
        # compute mean of activation
        self.x_mean = msops.mean(msops.abs(self.cat_samples), axis=0)
        logger.debug(f"AWQSmoothLinearCell: x_mean of Layer({self._layer_name}) is {{{self.x_mean.shape}, "
                     f"{self.x_mean.dtype}, {self.x_mean.asnumpy()}}}")

    def _search_best_scale(self, alpha):
        """search best scale"""
        best_scale = self._compute_best_scale(alpha)
        # pylint: disable=protected-access
        self.fp16_weight._offload()
        return best_scale

    def _compute_best_scale(self, alpha):
        """compute best scale"""
        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")

        group_size = self.cfg.group_size if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP \
              else self._layer.weight.shape[self.ic_axis]

        fp16_output = self._module_forward()

        for ratio in alpha:
            scales = self._calc_smooth_scale(ratio)
            self._apply_weight_smooth(scales)
            _, _, pesudo_weight = quant_tensor(self._layer.weight.data,
                                               self.w_quant_min,
                                               self.w_quant_max,
                                               self.cfg.weight_narrow_range,
                                               self.cfg.weight_symmetric,
                                               True,
                                               group_size,
                                               self.cfg.weight_quant_dtype,
                                               self.oc_axis,
                                               True,
                                               True)
            logger.debug(f"AWQSmoothLinearCell: search scale alpha {ratio}, pesudo weight of Layer({self._layer_name}) "
                         f"is {{{pesudo_weight.shape}, {pesudo_weight.dtype}, {pesudo_weight.asnumpy()}}}")
            weight_scales = msops.expand_dims(scales, 0)
            if not self._layer.transpose_b:
                weight_scales = weight_scales.transpose()
            self._layer.weight.set_data(msops.div(pesudo_weight, weight_scales))
            pseudo_output = self._module_forward()
            self._layer.weight.set_data(Tensor(self.fp16_weight.asnumpy(), dtype=self.compute_type))

            loss = msops.mse_loss(fp16_output, pseudo_output, reduction='mean')
            logger.info(f"AWQSmoothLinearCell: search scale alpha {ratio}, scale loss of Layer({self._layer_name}) "
                        f"is {{{loss.shape}, {loss.dtype}, {loss}}}")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales
        logger.info(f"AWQSmoothLinearCell: best scale alpha {best_ratio}, best_scale of Layer({self._layer_name}) "
                    f"is {{{best_scale.shape}, {best_scale.dtype}, {best_scale.asnumpy()}}}")
        if best_ratio == -1:
            raise ValueError(f"best_ratio=-1 is not correct, please check history of loss: {history}.")
        return best_scale

    def _attn_forward(self, samples):
        outputs = []
        for i, sample in enumerate(samples):
            output = self.forward_module(sample.expand_dims(0), self.kwargs[i]["batch_valid_length"],
                                         self.kwargs[i]["block_tables"], self.kwargs[i]["slot_mapping"],
                                         self.args[i][1], self.args[i][2], self.kwargs[i]["prefix_keys_values"])
            outputs.append(output.squeeze())
        return msops.cat(tuple(outputs), axis=0)

    def _module_forward(self):
        if "w_qkv" in self._layer_name:
            self.forward_module = self.decoder.attention
            return self._attn_forward(self.samples)
        if "w_gate_hidden" in self._layer_name:
            self.forward_module = self.decoder.feed_forward
        else:
            self.forward_module = self._layer
        return self.forward_module(self.cat_samples)

    def smooth(self):
        """smooth"""
        self._get_statistic_data()
        smooth_alpha = self.cfg.algo_args.get('smooth_alpha', [i/20 for i in range(20)])
        if isinstance(smooth_alpha, list):
            smooth_scale = self._search_best_scale(smooth_alpha)
        elif isinstance(smooth_alpha, float):
            smooth_scale = self._calc_smooth_scale(smooth_alpha)
        else:
            raise ValueError(f"AWQConfig smooth alpha only support list or float type, but got {type(smooth_alpha)}")
        self._apply_smooth(smooth_scale)

    def process(self):
        if not self.samples:
            raise RuntimeError("Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.smooth()
        # pylint: disable=protected-access
        self.layer.weight._offload()
        self.cat_samples = None
        self.samples.clear()
