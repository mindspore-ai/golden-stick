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
import gc

from mindspore import ops as msops
from mindspore import nn, Tensor, Parameter, mint
from mindspore import dtype as msdtype
import numpy as np
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm
import mindspore as ms
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoothQuant, LinearAutoSmoother
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.ptq.hal import ParallelType
from .linear_wrapper import WrapperLinearCell
from mindspore_gs.common.json_cache import JSONCache
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from typing import Optional


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        parallel_type = None
        self.transpose_b = True
        self.compute_type = self.layer.weight.dtype
        self.is_rowparallel = parallel_type == ParallelType.ROW_PARALLEL
        self.is_colparallel = parallel_type == ParallelType.COL_PARALLEL
        self.compute_range_kurtosis = False

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

    @staticmethod
    def kurtosis_score(x, epsilon=1e-10):
        mean = np.mean(x)
        std = np.std(x)
        z = (x - mean) / (std + epsilon)
        kurt = (np.mean(z ** 4) - 3)
        return float(kurt)

    @staticmethod
    def xrange(x):
        max_value = np.max(x)
        xmax = np.abs(max_value)
        min_value = np.min(x)
        xmin = np.abs(min_value)
        norm = np.maximum(xmax, xmin)
        std = np.std(x)

        if std == 0:
            return norm
        else:
            return norm / std

    def _apply_smooth(self, smooth_scale):
        """_apply_smooth"""
        self.layer.smooth_scale = Parameter(smooth_scale.astype(self.compute_type))
        if self.compute_range_kurtosis:
            smooth_sample = self.cat_samples / smooth_scale
            smooth_sample_fp = smooth_sample.astype(ms.float32)
            smooth_sample_np = smooth_sample_fp.asnumpy()
            logger.info(f"{self._layer_name}'s cat samples after apply smooth,"
                        f"kurtosis_score is:{self.kurtosis_score(smooth_sample_np)},"
                        f"xrange is:{self.xrange(smooth_sample_np)}.")

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


class SearchOutlierSuppressionLiteLinearCell(SmoothQuantLinearCell):
    """SearchOutlierSuppressionLiteLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE
        LinearAutoSmoother.reg_layer_map(nn.Dense, SearchOutlierSuppressionLiteLinearCell, SmoothChecker())
        LinearAutoSmoother.reg_layer_map(mint.nn.Linear, SearchOutlierSuppressionLiteLinearCell, SmoothChecker())

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.w_quant_max, self.w_quant_min = msops.max, msops.min
        self.x_quant_max, self.x_quant_min = msops.max, msops.min
        self.w_sum = msops.sum
        self.scale_max, self.scale_min = msops.max, msops.min
        rank = len(linear.weight.shape)
        self.ic_axis = rank - 1 if self._transpose_b() else rank - 2
        self.oc_axis = rank - 2 if self._transpose_b() else rank - 1
        self.oc = linear.weight.shape[self.oc_axis]
        self.is_expert = rank == 3
        self.expert_num = linear.weight.shape[0] if self.is_expert else -1
        if self.layer.has_bias and self.expert_num and self.expert_num > 0:
            raise ValueError(f"Only moe cell without bias is supported, but {linear_name} has bias.")

        self.decoder = kwargs.get("decoder_layer", None)
        self.args = kwargs.get("layer_args", None)
        self.kwargs = kwargs.get("layer_kwargs", None)

        self.x_scale_fast = None
        self.x_zp = None
        self.y_zp = None
        self.deq_scale = None
        self.quant_forward = False
        self.has_bias = self.layer.has_bias
        self.bias = None
        if self.has_bias:
            self.bias = self.layer.bias

        if "osl" in context.algorithm_cache_path:
            cache_file_path = os.path.join(context.algorithm_cache_path["osl"], f'rank_{context.rank_id}', \
                                           'osl_smooth.json')
        else:
            cache_file_path = ''
        self.cache: Optional[JSONCache] = JSONCache(cache_file_path)

    def _quant_info(self):
        return "OSL"

    def _search_best_scale(self, alpha):
        """search best scale"""
        best_alpha = self.cache.get(self.layer_name)
        if best_alpha:
            logger.info(f'layer {self.layer_name} using cached alpha: {best_alpha}')
            best_scale = self._calc_smooth_scale(best_alpha)
            logger.info(f'OSLLinearCell: best scale alpha {best_alpha} of Layer({self._layer_name}).'
                        ' Used cache.')
        else:
            best_scale, best_alpha = self._compute_best_scale(alpha)
            self.cache.put(self.layer_name, best_alpha)
        return best_scale

    def _expertwise_to_tokenwise(self, expertwise, group_list):
        indices = msops.arange(0, self.expert_num, dtype=msdtype.int32)
        indices = msops.repeat_interleave(indices, group_list)
        indices = msops.broadcast_to(indices, (self.oc, indices.shape[0])).t()
        return msops.gather_elements(expertwise, 0, indices)

    def construct(self, x, *args, **kwargs):
        """construct"""
        if self.quant_forward:
            x = x * self.x_scale_fast + self.x_zp
            x = msops.round(x)
            x = msops.clip(x, -128., 127.)
            if self.has_bias:
                self._layer.bias = None
                self._layer.has_bias = False

        x = x.astype(msdtype.float32)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.float32))
        y = self._layer(x, *args, **kwargs)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.bfloat16))

        if self.quant_forward:
            y_zp = self.y_zp
            deq_scale = self.deq_scale
            if self.is_expert:
                group_list = kwargs.get('group_list', None)
                if group_list is None:
                    group_list = args[0]
                y_zp = self._expertwise_to_tokenwise(y_zp, group_list)
                deq_scale = self._expertwise_to_tokenwise(deq_scale.squeeze(), group_list)
            y = (y - y_zp) * deq_scale
            y = msops.cast(y, self.compute_type)
            if self.has_bias:
                y = msops.add(y, self.bias)
                self._layer.bias = self.bias
                self._layer.has_bias = True
        y = y.astype(self.compute_type)
        return y

    @staticmethod
    def _x_var_mean(x):
        return msops.ReduceStd()(x)

    def xrange(self, x, minop, maxop):
        """xrange"""
        xmax = msops.abs(maxop(x)[0].reshape(-1)).asnumpy()
        xmin = msops.abs(minop(x)[0].reshape(-1)).asnumpy()
        norm = np.maximum(xmax, xmin)
        std = self._x_var_mean(x)
        return norm / std

    def check_xrange(self, xold, xnew):
        """check_xrange"""
        range_old = self.xrange(xold, self.x_quant_min, self.x_quant_max)
        range_new = self.xrange(xnew, self.x_quant_min, self.x_quant_max)
        logger.info(f"Range of {self.layer_name} before {range_old}, after {range_new}")

    def _compute_best_scale(self, alpha):
        """compute best scale"""
        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")
        fp16_weight = self._layer.weight.value()

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
                                            False,
                                            high_precision_params=False)
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
                                                False,
                                                high_precision_params=False)
            t_w_scale = Tensor(w_scale)
            if self._transpose_b():
                t_w_scale = msops.transpose(t_w_scale, (1, 0))
            self.x_scale_fast = Tensor(x_scale)
            self.deq_scale = msops.cast((self.x_scale_fast * t_w_scale), msdtype.float32)
            self.x_scale_fast = msops.cast(1 / (self.x_scale_fast * Tensor(scales)), msdtype.float32)
            self.x_zp = Tensor(x_zp)
            self._layer.weight.set_data(msops.cast(q_weight, self._layer.weight.dtype))
            self.y_zp = q_weight.sum(axis=self.ic_axis, dtype=msdtype.int32) * self.x_zp.astype(msdtype.int32)
            if self.is_rowparallel and self.context.tp_size > 1:
                self.y_zp = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(self.y_zp)
            quant_output = self._module_forward(True)
            msops.assign(self._layer.weight, fp16_weight)
            loss = self._loss(fp16_output, quant_output)
            logger.info(f"OSLLinearCell: search alpha {ratio}, loss of Layer({self._layer_name}) is {loss}.")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

            self.x_scale_fast = None
            self.deq_scale = None
            self.x_zp = None
            self.y_zp = None
            gc.collect()

        del fp16_weight
        del fp16_output
        del scales
        del xs
        del x_scale
        del x_zp
        del w_scale
        del q_weight
        del t_w_scale
        del quant_output
        gc.collect()
        if best_ratio == -1:
            raise RuntimeError(f"Found no suitablt ratio, please check history of loss: {history}.")
        logger.info(f"OSLLinearCell: best scale alpha {best_ratio}, best_error of Layer({self._layer_name}) "
                    f"is {best_error}")
        return best_scale, best_ratio

    def _module_forward(self, is_quant=False):
        self.quant_forward = is_quant
        results = []
        cnt = 0
        for args, kwargs in zip(self.args, self.kwargs):
            cnt += 1
            results.append(self.decoder(*args, **kwargs))
        self.quant_forward = False
        return results

    def _loss(self, preds, grounds):
        total_loss = 0
        for pred, ground in zip(preds, grounds):
            ground = msops.cast(ground[0], msdtype.float32)
            pred = msops.cast(pred[0], msdtype.float32)
            total_loss += float(msops.mse_loss(ground, pred, reduction='mean'))
        return total_loss / len(grounds)

    def smooth(self):
        """smooth"""
        smooth_alpha = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        smooth_scale = self._search_best_scale(smooth_alpha)
        self._apply_smooth(smooth_scale)

    def process(self):
        if not self.samples:
            raise RuntimeError("Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.smooth()
        self.cat_samples = None
        self.samples.clear()
        return self.layer


class QKVSuppressionLiteLinearCell(SearchOutlierSuppressionLiteLinearCell):

    @staticmethod
    def reg_self():
        """reg_self"""
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE
        LinearAutoSmoother.reg_qkv_ffn_map("qkv", QKVSuppressionLiteLinearCell, SmoothChecker())

    quant_forward = False
    smooth_scale = None
    x_scale_fast = None
    x_zp = None
    deq_scale = {}
    y_zp = {}

    def __init__(self, linear_name, linear, context, cfg, concat_weight, concat_linear_map, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.qkv_tag = linear_name.split('.')[-1]
        self.concat_weight = concat_weight
        self.concat_linear = concat_linear_map

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

        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.concat_weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.concat_weight, weight_smooth_minmax_axis)[0]))

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

    def _apply_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._transpose_b():
            weight_scale = msops.transpose(weight_scale, (1, 0))
        for qkv_tag, layer in self.concat_linear.items():
            orin_dtype = layer.weight.dtype
            weight = msops.mul(layer.weight, weight_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(layer.weight, weight)
            logger.debug(f"SmoothLinearCell: smoothed_weight of Layer({qkv_tag}) is {{{layer.weight.shape}, "
                        f"{layer.weight.dtype}}}")

    def _module_forward(self, is_quant=False):
        QKVSuppressionLiteLinearCell.quant_forward = is_quant
        results = []
        cnt = 0
        for args, kwargs in zip(self.args, self.kwargs):
            cnt += 1
            results.append(self.decoder(*args, **kwargs))
        QKVSuppressionLiteLinearCell.quant_forward = False
        return results

    def _compute_best_scale(self, alpha):
        """compute best scale"""
        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")
        fp16_weight = {layer_name: layer.weight.value() for layer_name, layer in self.concat_linear.items()}

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
                                            False,
                                            high_precision_params=False)
            for qkv_tag, layer in self.concat_linear.items():
                w_scale, _, q_weight = quant_tensor(layer.weight.data,
                                                    self.w_quant_min,
                                                    self.w_quant_max,
                                                    self.cfg.weight_narrow_range,
                                                    self.cfg.weight_symmetric,
                                                    False,
                                                    group_size,
                                                    self.cfg.weight_quant_dtype,
                                                    self.oc_axis,
                                                    True,
                                                    False,
                                                    high_precision_params=False)
                t_w_scale = Tensor(w_scale)
                if self._transpose_b():
                    t_w_scale = msops.transpose(t_w_scale, (1, 0))
                QKVSuppressionLiteLinearCell.x_scale_fast = Tensor(x_scale)
                QKVSuppressionLiteLinearCell.deq_scale[qkv_tag] = msops.cast((QKVSuppressionLiteLinearCell.x_scale_fast * t_w_scale), msdtype.float32)
                QKVSuppressionLiteLinearCell.x_scale_fast = msops.cast(1 / (QKVSuppressionLiteLinearCell.x_scale_fast * Tensor(scales)), msdtype.float32)
                QKVSuppressionLiteLinearCell.x_zp = Tensor(x_zp)
                layer.weight.set_data(msops.cast(q_weight, layer.weight.dtype))
                QKVSuppressionLiteLinearCell.y_zp[qkv_tag] = q_weight.sum(axis=self.ic_axis, dtype=msdtype.int32) * QKVSuppressionLiteLinearCell.x_zp.astype(msdtype.int32)
                if self.is_rowparallel and self.context.tp_size > 1:
                    QKVSuppressionLiteLinearCell.y_zp[qkv_tag] = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(QKVSuppressionLiteLinearCell.y_zp[qkv_tag])
            quant_output = self._module_forward(True)
            for qkv_tag, weight in fp16_weight.items():
                msops.assign(self.concat_linear[qkv_tag].weight, weight)
            loss = self._loss(fp16_output, quant_output)
            logger.info(f"OSLLinearCell: search alpha {ratio}, loss of Layer({self._layer_name}) is {loss}.")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

            QKVSuppressionLiteLinearCell.x_scale_fast = None
            QKVSuppressionLiteLinearCell.x_zp = None
            QKVSuppressionLiteLinearCell.deq_scale = {}
            QKVSuppressionLiteLinearCell.y_zp = {}
            gc.collect()

        del fp16_weight
        del fp16_output
        del scales
        del xs
        del x_scale
        del x_zp
        del w_scale
        del q_weight
        del t_w_scale
        del quant_output
        gc.collect()
        if best_ratio == -1:
            raise RuntimeError(f"Found no suitablt ratio, please check history of loss: {history}.")
        logger.info(f"OSLLinearCell: best scale alpha {best_ratio}, best_error of Layer({self._layer_name}) "
                    f"is {best_error}")
        return best_scale, best_ratio

    def construct(self, x, *args, **kwargs):
        """construct"""
        if QKVSuppressionLiteLinearCell.quant_forward:
            x = x * QKVSuppressionLiteLinearCell.x_scale_fast + QKVSuppressionLiteLinearCell.x_zp
            x = msops.round(x)
            x = msops.clip(x, -128., 127.)
            if self.has_bias:
                self._layer.bias = None
                self._layer.has_bias = False
        x = x.astype(msdtype.float32)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.float32))
        y = self._layer(x, *args, **kwargs)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.bfloat16))
        if QKVSuppressionLiteLinearCell.quant_forward:
            y_zp = QKVSuppressionLiteLinearCell.y_zp[self.qkv_tag]
            deq_scale = QKVSuppressionLiteLinearCell.deq_scale[self.qkv_tag]
            if self.is_expert:
                group_list = kwargs.get('group_list', None)
                if group_list is None:
                    group_list = args[0]
                y_zp = self._expertwise_to_tokenwise(y_zp, group_list)
                deq_scale = self._expertwise_to_tokenwise(deq_scale.squeeze(), group_list)
            y = (y - y_zp) * deq_scale
            y = msops.cast(y, self.compute_type)
            if self.has_bias:
                y = msops.add(y, self.bias)
                self._layer.bias = self.bia
                self._layer.has_bias = True
        y = y.astype(self.compute_type)
        return y

    def smooth(self):
        """smooth"""
        smooth_alpha = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        smooth_scale = self._search_best_scale(smooth_alpha)
        QKVSuppressionLiteLinearCell.smooth_scale = smooth_scale
        self._apply_smooth(smooth_scale)

    def process(self):
        if not self.samples:
            raise RuntimeError("Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        if "q_proj" in self._layer_name:
            self.smooth()
        else:
            assert QKVSuppressionLiteLinearCell.smooth_scale is not None
            logger.info(f"OSLLinearCell: apply smooth for Layer({self._layer_name}), use the same smooth_scale as q_proj.")
            self._apply_smooth(QKVSuppressionLiteLinearCell.smooth_scale)
        self.cat_samples = None
        self.samples.clear()
        return self.layer


class FFNSuppressionLiteLinearCell(SearchOutlierSuppressionLiteLinearCell):

    @staticmethod
    def reg_self():
        """reg_self"""
        class SmoothChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE
        LinearAutoSmoother.reg_qkv_ffn_map("ffn", FFNSuppressionLiteLinearCell, SmoothChecker())

    quant_forward = False
    smooth_scale = None
    x_scale_fast = None
    x_zp = None
    deq_scale = {}
    y_zp = {}
    def __init__(self, linear_name, linear, context, cfg, concat_weight, concat_linear_map, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.ffn_tag = linear_name.split('.')[-1]
        self.concat_weight = concat_weight
        self.concat_linear = concat_linear_map


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

        weight_max = msops.maximum(msops.abs(self.w_obs_max(self.concat_weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(self.w_obs_min(self.concat_weight, weight_smooth_minmax_axis)[0]))

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

    def _apply_weight_smooth(self, smooth_scale: Tensor):
        """_apply_weight_smooth"""
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._transpose_b():
            weight_scale = msops.transpose(weight_scale, (1, 0))
        for ffn_tag, layer in self.concat_linear.items():
            orin_dtype = layer.weight.dtype
            weight = msops.mul(layer.weight, weight_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(layer.weight, weight)
            logger.debug(f"SmoothLinearCell: smoothed_weight of Layer({ffn_tag}) is {{{layer.weight.shape}, "
                        f"{layer.weight.dtype}}}")

    def _module_forward(self, is_quant=False):
        FFNSuppressionLiteLinearCell.quant_forward = is_quant
        results = []
        cnt = 0
        for args, kwargs in zip(self.args, self.kwargs):
            cnt += 1
            results.append(self.decoder(*args, **kwargs))
        FFNSuppressionLiteLinearCell.quant_forward = False
        return results

    def _compute_best_scale(self, alpha):
        """compute best scale"""
        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")
        fp16_weight = {layer_name: layer.weight.value() for layer_name, layer in self.concat_linear.items()}

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
                                            False,
                                            high_precision_params=False)
            for ffn_tag, layer in self.concat_linear.items():
                w_scale, _, q_weight = quant_tensor(layer.weight.data,
                                                    self.w_quant_min,
                                                    self.w_quant_max,
                                                    self.cfg.weight_narrow_range,
                                                    self.cfg.weight_symmetric,
                                                    False,
                                                    group_size,
                                                    self.cfg.weight_quant_dtype,
                                                    self.oc_axis,
                                                    True,
                                                    False,
                                                    high_precision_params=False)
                t_w_scale = Tensor(w_scale)
                if self._transpose_b():
                    t_w_scale = msops.transpose(t_w_scale, (1, 0))
                FFNSuppressionLiteLinearCell.x_scale_fast = Tensor(x_scale)
                FFNSuppressionLiteLinearCell.deq_scale[ffn_tag] = msops.cast((FFNSuppressionLiteLinearCell.x_scale_fast * t_w_scale), msdtype.float32)
                FFNSuppressionLiteLinearCell.x_scale_fast = msops.cast(1 / (FFNSuppressionLiteLinearCell.x_scale_fast * Tensor(scales)), msdtype.float32)
                FFNSuppressionLiteLinearCell.x_zp = Tensor(x_zp)
                layer.weight.set_data(msops.cast(q_weight, layer.weight.dtype))
                FFNSuppressionLiteLinearCell.y_zp[ffn_tag] = q_weight.sum(axis=self.ic_axis, dtype=msdtype.int32) * FFNSuppressionLiteLinearCell.x_zp.astype(msdtype.int32)
                if self.is_rowparallel and self.context.tp_size > 1:
                    FFNSuppressionLiteLinearCell.y_zp[ffn_tag] = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(FFNSuppressionLiteLinearCell.y_zp[ffn_tag])
            quant_output = self._module_forward(True)
            for ffn_tag, weight in fp16_weight.items():
                msops.assign(self.concat_linear[ffn_tag].weight, weight)
            loss = self._loss(fp16_output, quant_output)
            logger.info(f"OSLLinearCell: search alpha {ratio}, loss of Layer({self._layer_name}) is {loss}.")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales

            FFNSuppressionLiteLinearCell.x_scale_fast = None
            FFNSuppressionLiteLinearCell.x_zp = None
            FFNSuppressionLiteLinearCell.deq_scale = {}
            FFNSuppressionLiteLinearCell.y_zp = {}
            gc.collect()

        del fp16_weight
        del fp16_output
        del scales
        del xs
        del x_scale
        del x_zp
        del w_scale
        del q_weight
        del t_w_scale
        del quant_output
        gc.collect()
        if best_ratio == -1:
            raise RuntimeError(f"Found no suitablt ratio, please check history of loss: {history}.")
        logger.info(f"OSLLinearCell: best scale alpha {best_ratio}, best_error of Layer({self._layer_name}) "
                    f"is {best_error}")
        return best_scale, best_ratio


    def construct(self, x, *args, **kwargs):
        """construct"""
        if FFNSuppressionLiteLinearCell.quant_forward:
            x = x * FFNSuppressionLiteLinearCell.x_scale_fast + FFNSuppressionLiteLinearCell.x_zp
            x = msops.round(x)
            x = msops.clip(x, -128., 127.)
            if self.has_bias:
                self._layer.bias = None
                self._layer.has_bias = False
        x = x.astype(msdtype.float32)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.float32))
        y = self._layer(x, *args, **kwargs)
        self._layer.weight = Parameter(self._layer.weight.astype(msdtype.bfloat16))
        if FFNSuppressionLiteLinearCell.quant_forward:
            y_zp = FFNSuppressionLiteLinearCell.y_zp[self.ffn_tag]
            deq_scale = FFNSuppressionLiteLinearCell.deq_scale[self.ffn_tag]
            if self.is_expert:
                group_list = kwargs.get('group_list', None)
                if group_list is None:
                    group_list = args[0]
                y_zp = self._expertwise_to_tokenwise(y_zp, group_list)
                deq_scale = self._expertwise_to_tokenwise(deq_scale.squeeze(), group_list)
            y = (y - y_zp) * deq_scale
            y = msops.cast(y, self.compute_type)
            if self.has_bias:
                y = msops.add(y, self.bias)
                self._layer.bias = self.bia
                self._layer.has_bias = True
        y = y.astype(self.compute_type)
        return y

    def smooth(self):
        """smooth"""
        smooth_alpha = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        smooth_scale = self._search_best_scale(smooth_alpha)
        FFNSuppressionLiteLinearCell.smooth_scale = smooth_scale
        self._apply_smooth(smooth_scale)

    def process(self):
        if not self.samples:
            raise RuntimeError("Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        if "gate_proj" in self.layer_name:
            self.smooth()
        else:
            assert FFNSuppressionLiteLinearCell.smooth_scale is not None
            logger.info(f"OSLLinearCell: apply smooth for Layer({self._layer_name}), use the same smooth_scale as gate_proj.")
            self._apply_smooth(FFNSuppressionLiteLinearCell.smooth_scale)
        self.cat_samples = None
        self.samples.clear()
        return self.layer