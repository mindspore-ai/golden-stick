# Copyright 2025 Huawei Technologies Co., Ltd
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

from mindspore import nn, mint
from mindspore import ops as msops

from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.algo_modules.mindone import LinearClipper
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.basic_functions.basic_quant_func import quant_tensor
from .linear_wrapper import WrapperLinearCell


class ClipLinearCell(WrapperLinearCell):
    """ClipLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class AutoClipChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.algo_args.get("weight_clip_ratio", [1 - i/20 for i in range(10)])

        LinearClipper.reg_layer_map(nn.Dense, ClipLinearCell, AutoClipChecker())
        LinearClipper.reg_layer_map(mint.nn.Linear, ClipLinearCell, AutoClipChecker())

    def _quant_info(self):
        return "wclip"

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.compute_type = self.layer.weight.dtype

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
            org_max_val = msops.abs(msops.max(w, axis=-1, keepdims=True)[0])
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
            org_max_val = msops.max(msops.abs(w), axis=-1, keepdims=True)[0]
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
                                         msops.min,
                                         msops.max,
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
                if i_b % 100 == 0:
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
        """process"""
        super().process()
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
