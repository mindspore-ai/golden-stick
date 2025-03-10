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
import numpy as np

from mindspore import Tensor, nn
from mindspore import ops as msops
from mindspore import dtype as msdtype
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import PTQMode, OutliersSuppressionType, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.hal import SmoothMatmul, SmoothMatmulForDeploy, OutlierSuppressionPlusMatmulForDeploy
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
    OSP = 4


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""

    def __init__(self, linear_name, linear, context, cfg, network_helper, **kwargs):
        super().__init__(linear_name, linear, context, cfg, network_helper, **kwargs)
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
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|input0_alpha", Tensor(alpha))
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|input1_activation_inputs",
                                  self.cat_samples)
        act_max = msops.maximum(msops.abs(self.x_obs_max(self.cat_samples, 0)[0]),
                                msops.abs(self.x_obs_min(self.cat_samples, 0)[0]))
        logger.debug(f"SmoothLinearCell: act_max of Layer({self._layer_name}) is {{{act_max.shape}, {act_max.dtype}, "
                     f"{act_max.asnumpy()}}}")
        input_max_pow = msops.pow(act_max, alpha)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|activation_minmax|output0_activation_minmax_pow",
                                  input_max_pow)
        weight_smooth_minmax_axis = -2 if self.layer.transpose_b else -1
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input0_alpha", Tensor(alpha))
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input1_weight", self.layer.weight)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale|weight_minmax|input2_weight_minmax_axis",
                                  Tensor(weight_smooth_minmax_axis))
        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.layer.weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.layer.weight, weight_smooth_minmax_axis)[0]))
        logger.debug(f"SmoothLinearCell: weight_max of Layer({self._layer_name}) is {{{weight_max.shape}, "
                     f"{act_max.dtype}, {weight_max.asnumpy()}}}")
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

    def _calc_awq_smooth_scale(self, alpha):
        """_calc_smooth_scale"""
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|input0_alpha", Tensor(alpha))
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|input1_activation_mean", self.x_mean)
        if self.cfg.algo_args.get("duo_scaling", True):
            x_pow = msops.pow(self.x_mean, alpha)
            w_pow = msops.pow(self.w_mean, 1 - alpha) + 1e-4
            self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|input2_weight_mean", self.w_mean)
            smooth_scale = (x_pow / w_pow).clamp(min=1e-4)
        else:
            smooth_scale = msops.pow(self.x_mean, alpha).clamp(1e-4).reshape(-1)
        minmax_norm = msops.sqrt(self.scale_max(smooth_scale)[0] * self.scale_min(smooth_scale)[0])
        smooth_scale = smooth_scale / minmax_norm
        smooth_scale[self.x_mean == 0] = 1
        smooth_scale[self.w_mean == 0] = 1
        logger.debug(f"AWQSmoothLinearCell: search scale alpha {alpha}, smooth scale of Layer({self._layer_name}) "
                     f"is {{{smooth_scale.shape}, {smooth_scale.dtype}, {smooth_scale.asnumpy()}}}")
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|output0_smooth_scale", smooth_scale)
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

    def _apply_group_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        org_shape = self._layer.weight.shape
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._layer.transpose_b:
            weight_scale = weight_scale.transpose()
            # [num_experts, ic, oc] -> [ic, num_experts * oc]
            weight = self._layer.weight.data.transpose((1, 0, 2)).reshape((org_shape[1], -1))
        else:
            # [num_experts, oc, ic] -> [num_experts * oc, ic]
            weight = self._layer.weight.data.reshape((-1, org_shape[-1]))

        orin_dtype = self._layer.weight.dtype
        weight = msops.mul(self._layer.weight, weight_scale)
        weight = self._layer.cast(weight, orin_dtype)

        if not self._layer.transpose_b:
            # [ic, num_experts * oc] -> [num_experts, ic, oc]
            weight = weight.reshape((org_shape[1], org_shape[0], org_shape[2])).transpose((1, 0, 2))
        else:
            # [num_experts * oc, ic] -> [num_experts, oc, ic]
            weight = weight.reshape(org_shape)
        self._layer.weight.set_data(weight)
        logger.debug(f"SmoothLinearCell: smoothed_group_weight of Layer({self._layer_name})"
                     f"is {{{self._layer.weight.shape}, "
                     f"{self._layer.weight.dtype}, {self._layer.weight.asnumpy()}}}")

    def _apply_act_smooth(self, smooth_scale: Tensor):
        """_apply_act_smooth"""
        if isinstance(self._layer.matmul, SmoothMatmul):
            self._layer.matmul.update(self._layer_name, self._layer.matmul.mm, smooth_scale)
        else:
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
        logger.info(f"apply smooth scale by infer mul op in {self._layer.matmul}.")

    def deploy(self):
        """deploy"""
        if self.cfg.mode == PTQMode.QUANTIZE or self.cfg.outliers_suppression == OutliersSuppressionType.NONE:
            return self.layer
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

    def _quant_info(self):
        return "SmoothQuant"


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

    def _quant_info(self) -> str:
        return ""

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


class SearchOmniQuantLinearCell(SmoothQuantLinearCell):
    """SearchOmniQuantLinearCell"""

    @staticmethod
    def reg_self():
        class SearchOmniQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OMNIQUANT_GRID

        LinearAutoSmoother.reg_layer_map(Linear, SearchOmniQuantLinearCell, SearchOmniQuantChecker())
        LinearAutoSmoother.reg_layer_map(ColumnParallelLinear, SearchOmniQuantLinearCell, SearchOmniQuantChecker())
        LinearAutoSmoother.reg_layer_map(RowParallelLinear, SearchOmniQuantLinearCell, SearchOmniQuantChecker())

    def __init__(self, linear_name, linear, context, cfg, network_helper, **kwargs):
        super().__init__(linear_name, linear, context, cfg, network_helper, **kwargs)
        if self.layer.has_bias:
            raise ValueError(f"Only cell without bias is supported, but {linear_name} has bias.")
        if isinstance(self.layer, Linear) and self.layer.activation_flag:
            raise ValueError(f"Only cell without activation is supported, but {linear_name} has activation.")

        if cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, False)
        else:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)
        self.x_quant_max, self.x_quant_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)
        self.w_sum = get_w_sum_op(cfg.tp_size, self.is_colparallel)
        self.scale_max, self.scale_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)

        rank = len(linear.weight.shape)
        self.ic_axis = rank - 1 if linear.transpose_b else rank - 2
        self.oc_axis = rank - 2 if linear.transpose_b else rank - 1
        self.oc = linear.weight.shape[self.oc_axis]
        self.fp16_weight = copy.deepcopy(self._layer.weight)

        self.decoder = kwargs.get("decoder_layer", None)
        self.args = kwargs.get("layer_args", None)
        self.kwargs = kwargs.get("layer_kwargs", None)

        self.x_scale_fast = None
        self.x_zp = None
        self.deq_scale = None
        self.quant_forward = False

    def _search_best_scale(self, alpha):
        """search best scale"""
        best_scale = self._compute_best_scale(alpha)
        # pylint: disable=protected-access
        self.fp16_weight._offload()
        return best_scale

    def construct(self, x, *args, **kwargs):
        if self.quant_forward:
            x = x * self.x_scale_fast + self.x_zp
            x = msops.round(x)
            x = msops.clip(x, -128., 127.)
        self._layer.compute_dtype = msdtype.float32
        y = self._layer(x, *args, **kwargs)
        self._layer.compute_dtype = self.compute_type
        if self.quant_forward:
            y_zp = (self.x_zp.astype("int32") * self._layer.weight.data.astype("int32")) \
                .sum(axis=1 if self._layer.transpose_b else 0) \
                .astype("int32")

            y = (y - y_zp) * self.deq_scale
            y = msops.cast(y, self.compute_type)
        return y

    @staticmethod
    def _x_var_mean(x):
        return msops.ReduceStd()(x)

    def xrange(self, x, minop, maxop):
        xmax = msops.abs(maxop(x)[0].reshape(-1)).asnumpy()
        xmin = msops.abs(minop(x)[0].reshape(-1)).asnumpy()
        norm = np.maximum(xmax, xmin)
        std = self._x_var_mean(x)
        return norm / std

    def check_xrange(self, xold, xnew):
        range_old = self.xrange(xold, self.x_quant_min, self.x_quant_max)
        range_new = self.xrange(xnew, self.x_quant_min, self.x_quant_max)
        logger.error(f"Range of {self.layer_name} before {range_old}, after {range_new}")

    def _compute_best_scale(self, alpha):
        """compute best scale"""
        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")

        group_size = self.cfg.group_size if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP \
              else self._layer.weight.shape[self.ic_axis]

        fp16_output = self._module_forward(False)

        for ratio in alpha:
            scales = self._calc_smooth_scale(ratio)
            self._apply_weight_smooth(scales)
            xs = self.cat_samples / scales
            x_scale, x_zp, _ = quant_tensor(xs,
                                            self.x_quant_min,
                                            self.x_quant_max,
                                            self.cfg.act_narrow_range,
                                            self.cfg.act_symmetric,
                                            False,
                                            group_size,
                                            self.cfg.act_quant_dtype,
                                            -1,
                                            False,
                                            False)
            w_scale, _, q_weight = quant_tensor(self._layer.weight.data,
                                                self.w_quant_min,
                                                self.w_quant_max,
                                                self.cfg.weight_narrow_range,
                                                self.cfg.weight_symmetric,
                                                False,
                                                group_size,
                                                self.cfg.weight_quant_dtype,
                                                self.oc_axis,
                                                True,
                                                False)
            t_w_scale = Tensor(w_scale)
            if self._layer.transpose_b:
                t_w_scale = msops.transpose(t_w_scale, (1, 0))
            self.x_scale_fast = Tensor(x_scale)
            self.deq_scale = msops.cast((self.x_scale_fast * t_w_scale), msdtype.float32)
            self.x_scale_fast = msops.cast(1 / (self.x_scale_fast * Tensor(scales)), msdtype.float32)
            self.x_zp = Tensor(x_zp)

            logger.debug(f"OmniSmoothLinearCell: search scale alpha {ratio}, pesudo weight of Layer({self._layer_name})"
                         f" is {{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
            self._layer.weight.set_data(msops.cast(q_weight, self.compute_type))
            quant_output = self._module_forward(True)
            self._layer.weight.set_data(Tensor(self.fp16_weight.asnumpy(), dtype=self.compute_type))

            loss = self._loss(fp16_output, quant_output)
            logger.info(f"OmniSmoothLinearCell: search scale alpha {ratio}, scale loss of Layer({self._layer_name}) "
                        f"is {{{loss.shape}, {loss.dtype}, {loss}}}")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales
        if best_ratio == -1:
            raise RuntimeError(f"Found no suitablt ratio, please check history of loss: {history}.")
        logger.info(f"OmniSmoothLinearCell: best scale alpha {best_ratio}, best_error of Layer({self._layer_name}) "
                    f"is {best_error}")
        return best_scale

    def _module_forward(self, is_quant=False):
        self.quant_forward = is_quant
        results = []
        for args, kwargs in zip(self.args, self.kwargs):
            results.append(self.decoder(*args, **kwargs))
        return results

    def _loss(self, preds, grounds):
        total_loss = np.array(0.0, np.float64)
        for pred, ground in zip(preds, grounds):
            ground = msops.cast(ground, msdtype.float32)
            pred = msops.cast(pred, msdtype.float32)
            total_loss += msops.mse_loss(ground, pred, reduction='mean')
        return total_loss / len(grounds)

    def smooth(self):
        """smooth"""
        smooth_alpha = [i/20 for i in range(21)]
        smooth_scale = self._search_best_scale(smooth_alpha)
        xs = self.cat_samples * smooth_scale
        self.check_xrange(self.cat_samples, xs)
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

    def __init__(self, linear_name, linear, context, cfg, network_helper, **kwargs):
        super().__init__(linear_name, linear, context, cfg, network_helper, **kwargs)

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

    def _quant_info(self):
        return "AWQ"

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
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|weight_mean|input0_weight", weight)
        w_max = msops.max(msops.abs(weight), self.ic_axis, keepdims=True)[0] + 1e-6
        w_scale = msops.abs(weight) / w_max
        w_scale = w_scale.reshape(org_shape)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|weight_mean|input1_weight_scale", w_scale)
        self.w_mean = self._get_mean_weight(w_scale, self.oc_axis)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|weight_mean|output0_weight_mean", self.w_mean)
        logger.debug(f"AWQSmoothLinearCell: w_mean of Layer({self._layer_name}) is {{{self.w_mean.shape}, "
                     f"{self.w_mean.dtype}, {self.w_mean.asnumpy()}}}")
        # compute mean of activation
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|activation_mean|input0_activation_inputs",
                                  self.cat_samples)
        self.x_mean = msops.mean(msops.abs(self.cat_samples), axis=0)
        self.cfg.dumper.dump_data(self.layer_name, "|smooth_scale_awq|activation_mean|output0_activation_mean",
                                  self.x_mean)
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
            if len(self._layer.weight.shape) == 3:
                self._apply_group_weight_smooth(scales)
            else:
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
                                               True,
                                               self._layer.transpose_b)
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
        """_module_forward"""
        if "routed_experts.ffn.w_gate_hidden" in self._layer_name:
            self.forward_module = self.decoder.feed_forward.routed_experts
            return self.forward_module(self.cat_samples.expand_dims(0))

        if "routed_experts.ffn.w1" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0), group_list=self.group_list)

        if "routed_experts.ffn.w3" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0), group_list=self.group_list)

        if "routed_experts.ffn.w2" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0), group_list=self.group_list)

        if "shared_experts.w_gate_hidden" in self._layer_name:
            self.forward_module = self.decoder.feed_forward.shared_experts
            return self.forward_module(self.cat_samples.expand_dims(0))

        if "shared_experts.w1" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0))

        if "shared_experts.w3" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0))

        if "shared_experts.w2" in self._layer_name:
            return self._layer(self.cat_samples.expand_dims(0))

        if "w_qkv" in self._layer_name:
            self.forward_module = self.decoder.attention
            return self._attn_forward(self.samples)

        if "w_gate_hidden" in self._layer_name:
            self.forward_module = self.decoder.feed_forward
        else:
            self.forward_module = self._layer
        return self.forward_module(self.cat_samples.expand_dims(0))

    def smooth(self):
        """smooth"""
        org_shape = self._layer.weight.shape
        if len(org_shape) == 3:
            if self._layer.transpose_b:
                # [num_experts, oc, ic] -> [num_experts * oc, ic]
                weight = self._layer.weight.data.reshape((-1, org_shape[-1]))
                self.ic_axis, self.oc_axis = 1, 0
            else:
                # [num_experts, ic, oc] -> [ic, num_experts * oc]
                weight = self._layer.weight.data.transpose((1, 0, 2)).reshape(org_shape[1], -1)
                self.ic_axis, self.oc_axis = 0, 1
                self.oc = weight.shape[1]
            self._layer.weight.set_data(weight)
        self._get_statistic_data()
        if len(org_shape) == 3:
            if self._layer.transpose_b:
                # [num_experts * oc, ic] -> [num_experts, oc, ic]
                weight = self._layer.weight.reshape(org_shape)
                self.ic_axis, self.oc_axis = 2, 1
            else:
                # [ic, num_experts * oc] -> [num_experts, ic, oc]
                weight = self._layer.weight.reshape((org_shape[1], org_shape[0],
                                                     org_shape[2])).transpose((1, 0, 2))
                self.ic_axis, self.oc_axis = 1, 2
            self._layer.weight.set_data(weight)
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


class OutlierSuppressionPlusLinearCell(AWQSmoothLinearCell):
    """SmoothQuantPlusLinearCell"""

    @staticmethod
    def reg_self():
        class OutlierSuppressionPlusChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS

        LinearAutoSmoother.reg_layer_map(Linear, OutlierSuppressionPlusLinearCell, OutlierSuppressionPlusChecker())
        LinearAutoSmoother.reg_layer_map(ColumnParallelLinear,
                                         OutlierSuppressionPlusLinearCell,
                                         OutlierSuppressionPlusChecker())
        LinearAutoSmoother.reg_layer_map(RowParallelLinear,
                                         OutlierSuppressionPlusLinearCell,
                                         OutlierSuppressionPlusChecker())
    def _quant_info(self):
        return "OutlierSuppressionPlus"

    def _get_smooth_method(self):
        return SmoothMethod.OSP

    def _apply_act_smooth_for_deploy(self, ic, compute_dtype):
        """_apply_act_smooth_by_insert_op_for_deploy"""
        self._layer.matmul = OutlierSuppressionPlusMatmulForDeploy.create(self._layer_name, self._layer.matmul, ic=ic,
                                                                          compute_dtype=compute_dtype)
