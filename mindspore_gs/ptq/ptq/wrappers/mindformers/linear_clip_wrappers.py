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
import gc
from types import MethodType

from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.algorithms.clipper import LinearClipper
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from .linear_wrapper import WrapperLinearCell
from .parallel_minmax import get_min_max_op


class ClipLinearCell(WrapperLinearCell):
    """ClipLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class AutoClipChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.algo_args.get("weight_clip_ratio", [1 - i/20 for i in range(10)])

        LinearClipper.reg_layer_map(Linear, ClipLinearCell, AutoClipChecker())
        LinearClipper.reg_layer_map(ColumnParallelLinear, ClipLinearCell, AutoClipChecker())
        LinearClipper.reg_layer_map(RowParallelLinear, ClipLinearCell, AutoClipChecker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            LinearClipper.reg_layer_map(ColumnParallelGroupLinear, ClipLinearCell, AutoClipChecker())
            LinearClipper.reg_layer_map(RowParallelGroupLinear, ClipLinearCell, AutoClipChecker())
            LinearClipper.reg_layer_map(ColumnParallelLinearWorldRegion, ClipLinearCell, AutoClipChecker())
            LinearClipper.reg_layer_map(RowParallelLinearWorldRegion, ClipLinearCell, AutoClipChecker())
        except ImportError:
            pass

    def _quant_info(self):
        return "wclip"

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.is_rowparallel = isinstance(self.layer, RowParallelLinear)
        self.is_colparallel = isinstance(self.layer, ColumnParallelLinear)
        self.is_linear = isinstance(self.layer, Linear)
        if not self.is_rowparallel and not self.is_colparallel and not self.is_linear:
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        self.compute_type = self.layer.dtype if self.is_linear else self.layer.compute_dtype

        self.w_obs_max, _ = get_min_max_op(cfg.tp_size, self.is_colparallel)
        if cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, False)
        else:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, self.is_rowparallel)

    def _calc_clip_val(self, weight_clip_ratio):
        """_calc_clip_val"""
        # [oc, ic]
        org_w_shape = self._layer.weight.shape

        group_size = self.cfg.group_size if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP \
              else org_w_shape[self.ic_axis]

        # [oc, 1, n_group, group_size]
        w = self._layer.weight.reshape(org_w_shape[0], 1, -1, group_size)
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        oc_batch_size = oc_batch_size if org_w_shape[0] % oc_batch_size == 0 else org_w_shape[0]

        w_all = w
        max_val_all = []
        for i_b in range(org_w_shape[0] // oc_batch_size):
            # [oc_batch_size, 1, n_group, group_size]
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
            # [oc, 1, n_group, 1]
            org_max_val = msops.abs(self.w_obs_max(w, axis=-1, keepdims=True)[0])
            max_val = msops.mul(org_max_val, weight_clip_ratio)
            max_val_all.append(max_val)
        max_val_all = msops.cat(max_val_all, axis=0)
        return max_val_all.squeeze(1)

    def _search_best_clip(self, weight_clip_ratio):
        """_search_best_clip"""

        n_sample_tokens = 512

        # [oc, ic]
        org_w_shape = self._layer.weight.shape

        group_size = self.cfg.group_size if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP \
              else self._layer.weight.shape[self.ic_axis]

        # [1, n_token, n_group, group_size]
        input_feat = self.cat_samples.reshape(1, self.cat_samples.shape[0], -1, group_size)

        step_size = max(1, input_feat.shape[1] // n_sample_tokens)

        # [1, n_sample_tokens, n_group, group_size]
        input_feat = input_feat[:, ::step_size]
        logger.debug(f"ClipLinearCell: input feature of Layer({self._layer_name}) is {{{input_feat.shape}, "
                     f"{input_feat.dtype}}}")

        # [oc, 1, n_group, group_size]
        w = self._layer.weight.reshape(org_w_shape[0], 1, -1, group_size)
        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64
        oc_batch_size = oc_batch_size if org_w_shape[0] % oc_batch_size == 0 else org_w_shape[0]

        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            # [oc_batch_size, 1, n_group, group_size]
            logger.info(f"ClipLinearCell: search iter {i_b}")
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]
            logger.debug(f"w of Layer({self._layer_name}) is {{{w.shape}, {w.dtype}}}")
            # [oc, 1, n_group, 1]
            org_max_val = self.w_obs_max(msops.abs(w), axis=-1, keepdims=True)[0]
            logger.debug(f"org_max_val of Layer({self._layer_name}) is {{{org_max_val.shape}, {org_max_val.dtype}}}")
            best_max_val = copy.deepcopy(org_max_val)
            min_errs = msops.ones_like(org_max_val).astype(w.dtype) * 1e9

            # [oc, n_sample_token, n_group]
            org_out = msops.sum(msops.mul(input_feat, w), dim=-1)

            for i_s in weight_clip_ratio:
                max_val = msops.mul(org_max_val, i_s)
                min_val = -max_val
                cur_w = msops.clamp(w, min_val, max_val)
                _, _, q_w = quant_tensor(cur_w,
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
                                         True)
                logger.debug(f"ClipLinearCell: search iter {i_b}, weight_clip_ratio {i_s}, "
                             f"pesudo weight of Layer({self._layer_name}) is {{{q_w.shape}, {q_w.dtype}}}")
                cur_out = msops.sum(msops.mul(input_feat, q_w), dim=-1)

                err = msops.mean(msops.pow(cur_out - org_out, 2), axis=1).reshape(min_errs.shape).astype(w.dtype)
                logger.info(f"Layer {self._layer_name}, weight clip search iter {i_b}, ratio {i_s}")
                logger.debug(f"clip err of Layer({self._layer_name}) is {{{err.shape}, {err.dtype}}}")
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)
        best_max_val = msops.cat(best_max_val_all, axis=0)
        del org_out
        del input_feat
        gc.collect()
        return best_max_val.squeeze(1)

    def _apply_clip(self, clip_val):
        """_apply_clip"""
        org_shape = self._layer.weight.shape
        weight = self._layer.weight.data.reshape(*clip_val.shape[:2], -1)
        weight = msops.clamp(weight, -clip_val, clip_val)
        weight = weight.reshape(org_shape)
        self._layer.weight.set_data(weight)
        logger.debug(f"ClipLinearCell: clip weight of Layer({self._layer_name}) is {{{weight.shape}, {weight.dtype}}}")

    def process(self):
        super(ClipLinearCell, self).process()
        org_shape = self._layer.weight.shape
        if len(org_shape) == 3:
            if self._layer.transpose_b:
                # [num_experts, oc, ic] -> [num_experts * oc, ic]
                weight = self._layer.weight.data.reshape((-1, org_shape[-1]))
            else:
                # [num_experts, ic, oc] -> [ic, num_experts * oc] -> [num_experts * oc, ic]
                weight = msops.transpose(self._layer.weight.data, (1, 0, 2)).reshape(org_shape[1], -1)
                weight = msops.transpose(weight, (1, 0))
            self._layer.weight.set_data(weight)
        # clip weight be [oc, ic] dims
        self.ic_axis, self.oc_axis = 1, 0
        weight_clip_ratio = self.cfg.algo_args.get("weight_clip_ratio", [1 - i/20 for i in range(10)])
        if isinstance(weight_clip_ratio, list):
            clip_val = self._search_best_clip(weight_clip_ratio)
        elif isinstance(weight_clip_ratio, float):
            clip_val = self._calc_clip_val(weight_clip_ratio)
        else:
            raise ValueError(f"AWQConfig clip alpha only support list or float type, but got {type(weight_clip_ratio)}")
        logger.debug(f"ClipLinearCell: best clip_val of Layer({self._layer_name}) is {{{clip_val.shape}, "
                     f"{clip_val.dtype}}}")
        self.cfg.dumper.dump_data(self.layer_name, "|awq_clip_val", clip_val)
        self._apply_clip(clip_val)

        if len(org_shape) == 3:
            if self.layer.transpose_b:
                weight = self._layer.weight.reshape(org_shape)
            else:
                weight = msops.transpose(self._layer.weight.data, (1, 0))
                weight = weight.reshape((org_shape[1], org_shape[0], org_shape[2]))
                weight = msops.transpose(weight, (1, 0, 2))
            self._layer.weight.set_data(weight)

    def deploy(self):
        """deploy"""
        if self.cfg.mode == PTQMode.QUANTIZE or not self.cfg.algo_args.get("apply_clip"):
            return self.layer
        logger.info("insert ops for smooth quant.")
        if self.is_colparallel:
            self.layer.sharded_state_dict = MethodType(ClipLinearCell.col_sharded_state_dict, self.layer)
        if self.is_rowparallel:
            self.layer.sharded_state_dict = MethodType(ClipLinearCell.row_sharded_state_dict, self.layer)
        return self.layer

    @staticmethod
    # pylint: disable=W0211
    def col_sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (self.tensor_parallel_group_size, 1) if self.transpose_b else (1, self.tensor_parallel_group_size)
        state_dict = {self.weight.name: {'shape': self.weight.shape, 'shard': w_shard}}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape, 'shard': (self.tensor_parallel_group_size,)}
        return state_dict

    @staticmethod
    #pylint: disable=W0211
    def row_sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        w_shard = (1, self.tensor_parallel_group_size) if self.transpose_b else (self.tensor_parallel_group_size, 1)
        state_dict = {self.weight.name: {'shape': self.weight.shape, 'shard': w_shard}}
        if self.has_bias:
            state_dict[self.bias.name] = {'shape': self.bias.shape,
                                          'shard': (1,)}
        return state_dict
