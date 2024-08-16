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

from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperLinearCell
from mindspore_gs.ptq.network_helpers import NetworkHelper, LayerType


class SmoothLinearCell(WrapperLinearCell):
    """SmoothLinearCell"""
    def __init__(self, linear_name: str, linear: Linear, cfg: InnerPTQConfig, net_helper: NetworkHelper):
        super().__init__(linear_name, linear, cfg)
        self.net_helper = net_helper

    def _calc_smooth_scale(self, alpha):
        """_calc_smooth_scale"""
        act_max = msops.maximum(msops.abs(msops.min(self.cat_samples, 0)[0]),
                                msops.abs(msops.max(self.cat_samples, 0)[0]))
        input_max_pow = msops.pow(act_max, alpha)
        weight_smooth_minmax_axis = -2 if self._linear.transpose_b else -1
        weight_max = msops.maximum(msops.abs(msops.min(self._linear.weight, weight_smooth_minmax_axis)[0]),
                                   msops.abs(msops.max(self._linear.weight, weight_smooth_minmax_axis)[0]))
        weight_max_pow = msops.pow(weight_max, 1 - alpha)
        smooth_scale = msops.div(input_max_pow, weight_max_pow).clamp(1e-5)
        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[input_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0
        return smooth_scale

    def _apply_smooth(self, smooth_scale):
        """_apply_smooth"""
        pre_layer = self.net_helper.get_pre_layer(self._linear_name)
        # pre-weight / scale
        if not pre_layer:
            raise ValueError("Not support inserting mul in x for smooth now, please enable qkv_concat and "
                             "ffn_concat.")
        if pre_layer.type_ == LayerType.NORM_LAYER:
            orin_dtype = pre_layer.layer.weight.dtype
            norm_weight = msops.div(pre_layer.layer.weight, smooth_scale)
            norm_weight = msops.cast(norm_weight, orin_dtype)
            msops.assign(pre_layer.layer.weight, norm_weight)
        if pre_layer.type_ == LayerType.LINEAR_LAYER:
            if isinstance(pre_layer.layer, Linear):
                linear: Linear = pre_layer.layer
            elif isinstance(pre_layer.layer, SmoothLinearCell):
                sqlinear: SmoothLinearCell = pre_layer.layer
                linear: Linear = sqlinear.linear
            else:
                raise RuntimeError(f"Got unexpected linear layer, name: {pre_layer.name} {pre_layer.layer}.")
            if linear.transpose_b:
                # oc * ic
                pre_scale = msops.expand_dims(smooth_scale, 1)
            else:
                # ic * oc
                pre_scale = msops.expand_dims(smooth_scale, 0)
            orin_dtype = linear.weight.dtype
            weight = msops.div(linear.weight, pre_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(linear.weight, weight)
        if pre_layer.type_ == LayerType.CONCAT_LINEAR_LAYER:
            if isinstance(pre_layer.layer, Linear):
                linear: Linear = pre_layer.layer
            elif isinstance(pre_layer.layer, SmoothLinearCell):
                sqlinear: SmoothLinearCell = pre_layer.layer
                linear: Linear = sqlinear.linear
            else:
                raise RuntimeError(f"Got unexpected linear layer, name: {pre_layer.name} {pre_layer.layer}.")
            if linear.transpose_b:
                # oc * ic
                oc = linear.weight.shape[0]
                pre_scale = msops.pad(smooth_scale, [oc - smooth_scale.shape[0], 0], value=1)
                pre_scale = msops.expand_dims(pre_scale, 1)
            else:
                # ic * oc
                oc = linear.weight.shape[1]
                pre_scale = msops.pad(smooth_scale, [oc - smooth_scale.shape[0], 0], value=1)
                pre_scale = msops.expand_dims(pre_scale, 0)
            orin_dtype = linear.weight.dtype
            weight = msops.div(linear.weight, pre_scale)
            weight = msops.cast(weight, orin_dtype)
            msops.assign(linear.weight, weight)
        # weight * scale
        weight_scale = msops.expand_dims(smooth_scale, 0)
        if not self._linear.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._linear.weight.dtype
        weight = msops.mul(self._linear.weight, weight_scale)
        weight = self._linear.cast(weight, orin_dtype)
        msops.assign(self._linear.weight, weight)

    def smooth(self, alpha=0.5):
        """smooth"""
        logger.info(f"Smooth linear {self.linear_name}")
        smooth_scale = self._calc_smooth_scale(alpha)
        self._apply_smooth(smooth_scale)

    def process(self):
        super(SmoothLinearCell, self).process()
        self.smooth(self.cfg.algo_args.get('alpha', 0.5))


class QuantLinearCell(WrapperLinearCell):
    """QuantLinearCell"""
    def __init__(self, linear_name: str, linear: Linear, cfg: InnerPTQConfig, net_helper: NetworkHelper):
        super().__init__(linear_name, linear, cfg)
        self.net_helper = net_helper
