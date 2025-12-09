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
"""anti-outliers algorithm."""
from typing import Tuple

from mindspore.nn import Cell
from mindspore import ops as msops
from mindspore import dtype as msdtype
from mindspore import Tensor

from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq.quant import PTQ
from mindspore_gs.ptq.basic_functions.processor import Processor
from mindspore_gs.ptq.context import InnerPTQConfig, OutliersSuppressionType
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell, Checker
from mindspore_gs.ptq.algo_modules.algo_module import AlgoModule


class LinearSmoother(AlgoModule):
    """LinearSmoother"""
    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """replace"""
        raise NotImplementedError

    def process(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """Outlier Suppression Smooth"""
        super().process(decoder_layer_name, decoder_layer, **kwargs)
        quant_model = kwargs.get('quant_model', None)
        if quant_model is None:
            raise ValueError("quant_model is required")
        search_inputs = kwargs.get('search_inputs', None)
        if search_inputs is None:
            raise ValueError("search_inputs is required")
        layers_info = quant_model.get_layers_for_smooth(decoder_layer)
        for layer_info in layers_info:
            self._check_layer_policies(decoder_layer_name, layer_info)
            if not isinstance(layer_info.curr_layer[0], QuantCell):
                continue
            self.is_gqa_wo_layer = (quant_model.num_attention_heads != quant_model.num_key_value_heads \
                 and '.o_proj' in layer_info.curr_layer[0].layer_name)
            self.weight_stats = self._get_weight_stats(layer_info,
                                                     quant_model.num_attention_heads,
                                                     quant_model.num_key_value_heads)
            self.act_stats = self._get_act_stats(layer_info,
                                               quant_model.num_attention_heads,
                                               quant_model.num_key_value_heads)
            smooth_scale = self._compute_smooth_scale(decoder_layer_name=decoder_layer_name,
                                                      decoder_layer=decoder_layer,
                                                      layer_info=layer_info,
                                                      search_inputs=search_inputs,
                                                      num_attention_heads=quant_model.num_attention_heads,
                                                      num_key_value_heads=quant_model.num_key_value_heads)
            layer_info.smooth_scale = smooth_scale

        # apply smooth scale to layers
        for layer_info in layers_info:
            if not isinstance(layer_info.curr_layer[0], QuantCell):
                continue
            self.is_gqa_wo_layer = (quant_model.num_attention_heads != quant_model.num_key_value_heads \
                 and '.o_proj' in layer_info.curr_layer[0].layer_name)
            self._apply_smooth_scale(layer_info,
                                     layer_info.smooth_scale,
                                     quant_model.num_attention_heads,
                                     quant_model.num_key_value_heads)
        # return to original layer
        self._transfrom_to_original_layer(decoder_layer)

    def _get_weight_stats(self, layer_info, num_attention_heads, num_key_value_heads):
        """_get_weight_stats"""
        # get current layers weight concat
        weight = []
        for layer in layer_info.curr_layer:
            weight.append(layer.layer.weight)
        weight = msops.concat(weight, axis=0)

        if self.is_gqa_wo_layer:
            weight_max = self._get_weight_stats_for_gqa(weight, num_attention_heads, num_key_value_heads)
        else:
            weight_max = msops.maximum(msops.abs(msops.max(weight, 0)[0]),
                                       msops.abs(msops.min(weight, 0)[0]))

        logger.info(f"weight_max of Layer({layer_info.curr_layer[0].layer_name}) "
                    f"is {{{weight_max.shape}, {weight_max.dtype}}}")
        return weight_max

    def _get_act_stats(self, layer_info, num_attention_heads, num_key_value_heads):
        """_get_act_stats"""
        act = layer_info.curr_layer[0].cat_samples

        if self.is_gqa_wo_layer:
            act_max = self._get_act_stats_for_gqa(act, num_attention_heads, num_key_value_heads)
        else:
            act_max = msops.maximum(
                msops.abs(msops.max(act, 0)[0]),
                msops.abs(msops.min(act, 0)[0]),
            )
        logger.info(f"act_max of Layer({layer_info.curr_layer[0].layer_name}) "
                    f"is {{{act_max.shape}, {act_max.dtype}}}")
        return act_max

    def _get_weight_stats_for_gqa(self, weight, num_attention_heads, num_key_value_heads):
        """_get_weight_stats_for_gqa"""
        num_groups = num_attention_heads // num_key_value_heads
        weight = weight.reshape(weight.shape[0], num_key_value_heads, num_groups, -1)

        weight_max = msops.max(msops.max(weight, axis=0)[0], axis=1)[0].reshape(-1,)
        weight_min = msops.min(msops.min(weight, axis=0)[0], axis=1)[0].reshape(-1,)
        return msops.maximum(msops.abs(weight_max), msops.abs(weight_min))

    def _get_act_stats_for_gqa(self, act, num_attention_heads, num_key_value_heads):
        """_get_act_stats_for_gqa"""
        num_groups = num_attention_heads // num_key_value_heads
        act = act.reshape(act.shape[0], num_key_value_heads, num_groups, -1)

        act_max = msops.max(msops.max(act, axis=0)[0], axis=1)[0].reshape(-1,)
        act_min = msops.min(msops.min(act, axis=0)[0], axis=1)[0].reshape(-1,)
        return msops.maximum(msops.abs(act_max), msops.abs(act_min))

    def _expand_scales_for_gqa_curr_layer(self, smooth_scale, num_attention_heads, num_key_value_heads):
        """_expand_scales_for_gqa"""
        copy_smooth_scale = smooth_scale.split(smooth_scale.shape[-1] // num_key_value_heads)
        updated_scales = []
        repeat_num = num_attention_heads // num_key_value_heads
        for idx in range(num_key_value_heads):
            updated_scales.append(msops.tile(copy_smooth_scale[idx], (repeat_num,)))
        return msops.concat(updated_scales)

    def _reduce_scales_for_gqa_prev_layer(self, smooth_scale, num_attention_heads, num_key_value_heads):
        """_reduce_scales_for_gqa_prev_layer"""
        # Assuming heads for K V activations are reduce with following pattern:
        # [h1, ... , h1, h2,..., h2, h3, ..., h3, h4, ..., h4] -> [h1, h2, h3, h4]
        reduced_num = num_attention_heads // num_key_value_heads
        copy_smooth_scale = smooth_scale.split(smooth_scale.shape[-1] // num_key_value_heads)

        reduced_scales = []
        scale_size = smooth_scale.shape[-1] // num_key_value_heads // reduced_num
        for idx in range(num_key_value_heads):
            reduced_scales.append(copy_smooth_scale[idx][:scale_size, ...])
        reduced_scales = msops.concat(reduced_scales)
        return msops.div(1, reduced_scales)

    def _compute_smooth_scale(self, **kwargs):
        """_compute_smooth_scale"""
        raise NotImplementedError

    def _calculate_smooth_scale(self, act_stats, weight_stats, ratio,
                                num_attention_heads, num_key_value_heads):
        """_calculate_smooth_scale"""
        act_max_pow = msops.pow(act_stats, ratio)
        weight_max_pow = msops.pow(weight_stats, 1 - ratio)
        smooth_scale = msops.div(act_max_pow, weight_max_pow).clamp(1e-5)
        # set 0 or nan to 1.0 to avoid quantization error
        smooth_scale[act_max_pow == 0] = 1.0
        smooth_scale[weight_max_pow == 0] = 1.0

        if self.is_gqa_wo_layer:
            smooth_scale = self._expand_scales_for_gqa_curr_layer(smooth_scale,
                                                                  num_attention_heads,
                                                                  num_key_value_heads)
        return smooth_scale

    def _apply_smooth_scale(self, layer_info, smooth_scale, num_attention_heads, num_key_value_heads):
        """_apply_smooth_scale"""
        if self.is_gqa_wo_layer:
            curr_layer_scale = smooth_scale
            prev_layer_scale = self._reduce_scales_for_gqa(smooth_scale,
                                                           num_attention_heads,
                                                           num_key_value_heads)
        else:
            curr_layer_scale = smooth_scale
            prev_layer_scale = msops.div(1, smooth_scale)

        if 'norm' not in type(layer_info.prev_layer).__name__.lower():
            prev_layer_scale = prev_layer_scale.expand_dims(-1)

        self._apply_smooth_scale_to_prev_layer(layer_info.prev_layer, prev_layer_scale)
        for layer in layer_info.curr_layer:
            self._apply_smooth_scale_to_curr_layer(layer, curr_layer_scale)

    def _reduce_scales_for_gqa(self, smooth_scale, num_attention_heads, num_key_value_heads):
        """_reduce_scales_for_gqa"""
        # Assuming heads for K V activations are reduce with following pattern:
        # [h1, ... , h1, h2,..., h2, h3, ..., h3, h4, ..., h4] -> [h1, h2, h3, h4]
        reduced_num = num_attention_heads // num_key_value_heads
        copy_smooth_scale = smooth_scale.split(smooth_scale.shape[-1] // num_key_value_heads)

        reduced_scales = []
        scale_size = smooth_scale.shape[-1] // num_key_value_heads // reduced_num
        for idx in range(num_key_value_heads):
            reduced_scales.append(copy_smooth_scale[idx][:scale_size, ...])
        reduced_scales = msops.concat(reduced_scales)
        return msops.div(1, reduced_scales)

    def _apply_smooth_scale_to_prev_layer(self, layer: Cell, smooth_scale: Tensor):
        """_apply_smooth_scale_to_curr_layer"""
        layer = layer.layer if isinstance(layer, QuantCell) else layer
        # only apply smooth scale to the last dim of the weight
        scale_len = smooth_scale.shape[0]
        new_weight = layer.weight.clone()
        new_weight[-scale_len:, ...] = msops.mul(new_weight[-scale_len:, ...], smooth_scale)
        layer.weight.set_data(new_weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            new_bias = layer.bias.clone()
            new_bias[-scale_len:] = msops.mul(new_bias[-scale_len:], smooth_scale.squeeze())
            layer.bias.set_data(new_bias)

    def _apply_smooth_scale_to_curr_layer(self, layer: Cell, smooth_scale: Tensor):
        """_apply_smooth_scale_to_curr_layer"""
        layer = layer.layer if isinstance(layer, QuantCell) else layer
        layer.weight.set_data(msops.mul(layer.weight, smooth_scale))

    def _transfrom_to_original_layer(self, decoder_layer: Cell):
        """transform to original layer"""
        class Replacer(Processor):
            """Replacer"""
            # pylint: disable=unused-argument
            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not isinstance(cell, QuantCell):
                    return cell, False
                return cell.layer, True
        Replacer().process(decoder_layer)

    def _check_layer_policies(self, decoder_layer_name, layer_info):
        """_check_layer_policies"""
        if len(layer_info.curr_layer) == 1:
            return True

        # check if all layers are either all QuantCell or all non-QuantCell
        first_layer = layer_info.curr_layer[0]
        is_first_layer_quant_cell = isinstance(first_layer, QuantCell)
        for layer in layer_info.curr_layer[1:]:
            is_layer_quant_cell = isinstance(layer, QuantCell)
            if is_layer_quant_cell != is_first_layer_quant_cell:
                raise ValueError(f"{decoder_layer_name}: Layers {layer} have different policy "
                                 f"with {first_layer}, please check layer policies of layers.")

        # if all layers are non-QuantCell, return True
        if not is_first_layer_quant_cell:
            return True

        # check if all layers have the same layer policy
        first_layer_policy = self.get_layer_policy(first_layer.layer_name)
        for layer in layer_info.curr_layer[1:]:
            layer_policy = self.get_layer_policy(layer.layer_name)
            if layer_policy != first_layer_policy:
                raise ValueError(f"Layers {layer.layer_name} have different layer policies with "
                                 f"{first_layer.layer_name}, please check layer policies of layers.")
        return True


class SmoothQuantSmoother(LinearSmoother):
    """SmoothQuantSmoother"""
    linear_map = {}
    fake_quant_linear_map = {}

    @staticmethod
    def reg_self():
        """register self"""
        # Check if already registered to avoid duplicate registration
        if SmoothQuantSmoother not in PTQ.pipeline:
            PTQ.pipeline.append(SmoothQuantSmoother)
        logger.info(f"Register algo_module {SmoothQuantSmoother} to {PTQ.__name__} pipeline.")
        # Add layer types that are not already in target_layer_type
        new_layer_types = tuple(set(SmoothQuantSmoother.linear_map.keys()) - set(PTQ.target_layer_type))
        if new_layer_types:
            PTQ.target_layer_type += new_layer_types

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} " \
                    f"to SmoothQuantSmoother, checker: {checker}")
        if not SmoothQuantSmoother.linear_map.get(layer_type):
            SmoothQuantSmoother.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            SmoothQuantSmoother.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = SmoothQuantSmoother.linear_map.get(layer_type) if not self.is_fake_quant else \
            SmoothQuantSmoother.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """infer_and_cache"""

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not SmoothQuantSmoother.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or
                        layer_policy.outliers_suppression != OutliersSuppressionType.SMOOTH):
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                return wrapper_cell, True

        Replacer(self).process(decoder_layer, decoder_layer_name)

    def _compute_smooth_scale(self, **kwargs):
        """_compute_smooth_scale"""
        num_attention_heads = kwargs.get('num_attention_heads', None)
        if num_attention_heads is None:
            raise ValueError("SmoothQuantSmoother: when compute smooth scale num_attention_heads is required.")
        num_key_value_heads = kwargs.get('num_key_value_heads', None)
        if num_key_value_heads is None:
            raise ValueError("SmoothQuantSmoother: when compute smooth scale num_key_value_heads is required.")
        return self._calculate_smooth_scale(self.act_stats,
                                            self.weight_stats,
                                            ratio=0.5,
                                            num_attention_heads=num_attention_heads,
                                            num_key_value_heads=num_key_value_heads)


class OSLSmoother(LinearSmoother):
    """Outlier Suppression Lite smoother"""

    linear_map = {}
    fake_quant_linear_map = {}

    @staticmethod
    def reg_self():
        """register self"""
        # Check if already registered to avoid duplicate registration
        if OSLSmoother not in PTQ.pipeline:
            PTQ.pipeline.append(OSLSmoother)
        logger.info(f"Register algo_module {OSLSmoother} to {PTQ.__name__} pipeline.")
        # Add layer types that are not already in target_layer_type
        new_layer_types = tuple(set(OSLSmoother.linear_map.keys()) - set(PTQ.target_layer_type))
        if new_layer_types:
            PTQ.target_layer_type += new_layer_types

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} " \
                    f"to OSLSmoother, checker: {checker}")
        if not OSLSmoother.linear_map.get(layer_type):
            OSLSmoother.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            OSLSmoother.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = OSLSmoother.linear_map.get(layer_type) if not self.is_fake_quant else \
            OSLSmoother.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """infer_and_cache"""

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not OSLSmoother.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or
                        layer_policy.outliers_suppression != OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE):
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                return wrapper_cell, True

        Replacer(self).process(decoder_layer, decoder_layer_name)

    def _compute_smooth_scale(self, **kwargs):
        """_compute_smooth_scale"""
        decoder_layer_name = kwargs.get('decoder_layer_name', None)
        if decoder_layer_name is None:
            raise ValueError("OSLSmoother: when compute smooth scale decoder_layer_name is required.")
        decoder_layer = kwargs.get('decoder_layer', None)
        if decoder_layer is None:
            raise ValueError("OSLSmoother: when compute smooth scale decoder_layer is required.")
        layer_info = kwargs.get('layer_info', None)
        if layer_info is None:
            raise ValueError("OSLSmoother: when compute smooth scale layer_info is required.")
        search_inputs = kwargs.get('search_inputs', None)
        if search_inputs is None:
            raise ValueError("OSLSmoother: when compute smooth scale search_inputs is required.")
        num_attention_heads = kwargs.get('num_attention_heads', None)
        if num_attention_heads is None:
            raise ValueError("OSLSmoother: when compute smooth scale num_attention_heads is required.")
        num_key_value_heads = kwargs.get('num_key_value_heads', None)
        if num_key_value_heads is None:
            raise ValueError("OSLSmoother: when compute smooth scale num_key_value_heads is required.")
        best_scale, _ = self._search_best_ratio(decoder_layer_name,
                                                decoder_layer,
                                                layer_info,
                                                search_inputs,
                                                num_attention_heads,
                                                num_key_value_heads)
        return best_scale

    def _search_best_ratio(self, decoder_layer_name, decoder_layer, layer_info, search_inputs,
                           num_attention_heads, num_key_value_heads):
        """_search_best_ratio"""
        fp_output = self._module_forward(decoder_layer, search_inputs)
        layer_policy = self.get_layer_policy(decoder_layer_name)
        smooth_alpha = layer_policy.algo_args.get('smooth_alpha', [i/20 for i in range(20)])
        # when GQA with o_proj layer, need expand smooth scale for curr_layer

        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")
        for ratio in smooth_alpha:
            scales = self._calculate_smooth_scale(self.act_stats, self.weight_stats, ratio,
                                                  num_attention_heads, num_key_value_heads)
            for layer in layer_info.curr_layer:
                layer.quant_forward = True
                layer.set_smooth_scale(scales)
            # calculate pseudo output
            pseudo_output = self._module_forward(decoder_layer, search_inputs)
            for layer in layer_info.curr_layer:
                layer.quant_forward = False
            loss = self._loss(fp_output, pseudo_output)
            logger.info(f"OSLSmoother: search scale alpha {ratio}, "
                        f"scale loss of Layer({decoder_layer_name}): {loss}")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales
        if isinstance(best_scale, Tensor):
            logger.info(f"OSLSmoother: best scale alpha {best_ratio}, "
                        f"best_scale of Layer({decoder_layer_name}) "
                        f"is {{{best_scale.shape}, {best_scale.dtype}}}")
        else:
            logger.info(f"OSLSmoother: best scale alpha {best_ratio}, "
                        f"best_scale of Layer({decoder_layer_name}): {best_scale}")
        if best_ratio == -1:
            raise ValueError(f"best_ratio=-1 is not correct, please check history of loss: {history}.")
        return best_scale, best_ratio

    def _module_forward(self, decoder_layer, search_inputs):
        """_module_forward"""
        results = []
        for args, kwargs in zip(search_inputs.layer_args, search_inputs.layer_kwargs):
            results.append(decoder_layer(*args, **kwargs))
        return results

    def _loss(self, preds, grounds):
        total_loss = 0
        for pred, ground in zip(preds, grounds):
            ground = msops.cast(ground[0], msdtype.float32)
            pred = msops.cast(pred[0], msdtype.float32)
            total_loss += float(msops.mse_loss(ground, pred, reduction='mean'))
        return total_loss / len(grounds)


class AWQSmoother(LinearSmoother):
    """AWQ smoother"""

    linear_map = {}
    fake_quant_linear_map = {}

    @staticmethod
    def reg_self():
        """register self"""
        # Check if already registered to avoid duplicate registration
        if AWQSmoother not in PTQ.pipeline:
            PTQ.pipeline.append(AWQSmoother)
        logger.info(f"Register algo_module {AWQSmoother} to {PTQ.__name__} pipeline.")
        # Add layer types that are not already in target_layer_type
        new_layer_types = tuple(set(AWQSmoother.linear_map.keys()) - set(PTQ.target_layer_type))
        if new_layer_types:
            PTQ.target_layer_type += new_layer_types

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} " \
                    f"to AWQSmoother, checker: {checker}")
        if not AWQSmoother.linear_map.get(layer_type):
            AWQSmoother.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            AWQSmoother.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = AWQSmoother.linear_map.get(layer_type) if not self.is_fake_quant else \
            AWQSmoother.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """infer_and_cache"""

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not AWQSmoother.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or
                        layer_policy.outliers_suppression != OutliersSuppressionType.AWQ):
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                return wrapper_cell, True

        Replacer(self).process(decoder_layer, decoder_layer_name)

    def _compute_smooth_scale(self, **kwargs):
        """_compute_smooth_scale"""
        decoder_layer_name = kwargs.get('decoder_layer_name', None)
        if decoder_layer_name is None:
            raise ValueError("OSLSmoother: when compute smooth scale decoder_layer_name is required.")
        decoder_layer = kwargs.get('decoder_layer', None)
        if decoder_layer is None:
            raise ValueError("OSLSmoother: when compute smooth scale decoder_layer is required.")
        layer_info = kwargs.get('layer_info', None)
        if layer_info is None:
            raise ValueError("OSLSmoother: when compute smooth scale layer_info is required.")
        search_inputs = kwargs.get('search_inputs', None)
        if search_inputs is None:
            raise ValueError("OSLSmoother: when compute smooth scale search_inputs is required.")
        num_attention_heads = kwargs.get('num_attention_heads', None)
        if num_attention_heads is None:
            raise ValueError("OSLSmoother: when compute smooth scale num_attention_heads is required.")
        num_key_value_heads = kwargs.get('num_key_value_heads', None)
        if num_key_value_heads is None:
            raise ValueError("OSLSmoother: when compute smooth scale num_key_value_heads is required.")
        best_scale, _ = self._search_best_ratio(decoder_layer_name,
                                                decoder_layer,
                                                layer_info,
                                                search_inputs,
                                                num_attention_heads,
                                                num_key_value_heads)
        return best_scale

    def _get_weight_stats(self, layer_info, num_attention_heads, num_key_value_heads):
        """_get_weight_stats"""
        # get current layers weight concat
        weight = []
        for layer in layer_info.curr_layer:
            weight.append(layer.layer.weight)
        weight = msops.concat(weight, axis=0)

        layer_policy = self.get_layer_policy(layer_info.curr_layer[0].layer_name)
        group_size = layer_policy.group_size
        org_shape = weight.shape
        if group_size > 0:
            weight = weight.reshape(-1, group_size)
        w_max = msops.max(msops.abs(weight), -1, keepdims=True)[0] + 1e-6
        weight = msops.abs(weight) / w_max.reshape(-1, 1)
        weight = weight.reshape(org_shape)

        if self.is_gqa_wo_layer:
            w_mean = self._get_weight_stats_for_gqa(weight, num_attention_heads, num_key_value_heads)
        else:
            w_mean = msops.mean(weight, axis=0)

        logger.info(f"AWQSmoother: weight_mean of Layer({layer_info.curr_layer[0].layer_name}) "
                    f"is {{{w_mean.shape}, {w_mean.dtype}}}")
        return w_mean

    def _get_act_stats(self, layer_info, num_attention_heads, num_key_value_heads):
        """_get_act_stats"""
        act = layer_info.curr_layer[0].cat_samples

        if self.is_gqa_wo_layer:
            x_mean = self._get_act_stats_for_gqa(act, num_attention_heads, num_key_value_heads)
        else:
            x_mean = msops.mean(msops.abs(act), axis=0)
        logger.info(f"AWQSmoother: act_mean of Layer({layer_info.curr_layer[0].layer_name}) "
                    f"is {{{x_mean.shape}, {x_mean.dtype}}}")
        return x_mean

    def _get_weight_stats_for_gqa(self, weight, num_attention_heads, num_key_value_heads):
        """_get_weight_stats_for_gqa"""
        num_groups = num_attention_heads // num_key_value_heads
        weight = weight.reshape(weight.shape[0], num_key_value_heads, num_groups, -1)

        w_mean = msops.mean(msops.mean(weight, axis=0), axis=1).reshape(-1,)
        return w_mean

    def _get_act_stats_for_gqa(self, act, num_attention_heads, num_key_value_heads):
        """_get_act_stats_for_gqa"""
        num_groups = num_attention_heads // num_key_value_heads
        act = act.reshape(act.shape[0], num_key_value_heads, num_groups, -1)

        x_mean = msops.mean(msops.mean(msops.abs(act), axis=0), axis=1).reshape(-1,)
        return x_mean

    # pylint: disable=arguments-differ
    def _calculate_smooth_scale(self, act_stats, weight_stats, ratio,
                                num_attention_heads, num_key_value_heads, layer_info):
        """_calculate_smooth_scale"""
        layer_policy = self.get_layer_policy(layer_info.curr_layer[0].layer_name)
        is_duo_scaling = layer_policy.algo_args.get("duo_scaling", True)
        if is_duo_scaling:
            x_pow = msops.pow(act_stats, ratio)
            w_pow = msops.pow(weight_stats, 1 - ratio) + 1e-4
            smooth_scale = (x_pow / w_pow).clamp(min=1e-4)
        else:
            smooth_scale = msops.pow(act_stats, ratio).clamp(1e-4).reshape(-1)

        smooth_scale[msops.isnan(smooth_scale.astype(msdtype.float32))] = 1
        minmax_norm = msops.sqrt(msops.max(smooth_scale)[0] * msops.min(smooth_scale)[0])
        smooth_scale = smooth_scale / minmax_norm
        smooth_scale[act_stats == 0] = 1
        smooth_scale[weight_stats == 0] = 1

        if self.is_gqa_wo_layer:
            smooth_scale = self._expand_scales_for_gqa_curr_layer(smooth_scale,
                                                                  num_attention_heads,
                                                                  num_key_value_heads)
        return smooth_scale

    def _search_best_ratio(self, decoder_layer_name, decoder_layer, layer_info, search_inputs,
                           num_attention_heads, num_key_value_heads):
        """_search_best_ratio"""
        fp_output = self._module_forward(decoder_layer, search_inputs)
        layer_policy = self.get_layer_policy(decoder_layer_name)
        smooth_alpha = layer_policy.algo_args.get('smooth_alpha', [i/20 for i in range(20)])
        # when GQA with o_proj layer, need expand smooth scale for curr_layer

        history = []
        best_ratio = -1
        best_scale = 0
        best_error = float("inf")
        for ratio in smooth_alpha:
            scales = self._calculate_smooth_scale(self.act_stats,
                                                  self.weight_stats,
                                                  ratio,
                                                  num_attention_heads,
                                                  num_key_value_heads,
                                                  layer_info)
            for layer in layer_info.curr_layer:
                layer.quant_forward = True
                layer.set_smooth_scale(scales)
            # calculate pseudo output
            pseudo_output = self._module_forward(decoder_layer, search_inputs)
            for layer in layer_info.curr_layer:
                layer.quant_forward = False
            loss = self._loss(pseudo_output, fp_output)
            logger.info(f"AWQSmoother: search scale alpha {ratio}, "
                        f"scale loss of Layer({decoder_layer_name}): {loss}")
            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scale = scales
        if isinstance(best_scale, Tensor):
            logger.info(f"AWQSmoother: best scale alpha {best_ratio}, "
                        f"best_scale of Layer({decoder_layer_name}) "
                        f"is {{{best_scale.shape}, {best_scale.dtype}}}")
        else:
            logger.info(f"AWQSmoother: best scale alpha {best_ratio}, "
                        f"best_scale of Layer({decoder_layer_name}): {best_scale}")
        if best_ratio == -1:
            raise ValueError(f"best_ratio=-1 is not correct, please check history of loss: {history}.")
        return best_scale, best_ratio

    def _module_forward(self, decoder_layer, search_inputs):
        """_module_forward"""
        results = []
        for args, kwargs in zip(search_inputs.layer_args, search_inputs.layer_kwargs):
            results.append(decoder_layer(*args, **kwargs))
        return results

    def _loss(self, preds, grounds):
        total_loss = 0
        for pred, ground in zip(preds, grounds):
            ground = msops.cast(ground[0], msdtype.float32)
            pred = msops.cast(pred[0], msdtype.float32)
            total_loss += float(msops.mse_loss(ground, pred, reduction='mean'))
        return total_loss / len(grounds)
