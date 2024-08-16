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
"""OmniQuant algorithm."""
import os
from functools import partial
import tqdm

import mindspore as ms
from mindspore.nn import Cell, MSELoss
from mindspore import ops as msops
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_network
from mindspore_gs.ptq import PTQMode
from mindspore_gs.ptq.processor import network_replace
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQApproach
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.omni_quant.quant_cells import SQLinearWrapper, SQLinearDeploy


class InputCatcher(Cell):
    """input catcher"""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        if hasattr(handler, "attention"):
            self.attention = handler.attention
        self.args = []
        self.kwargs = []

    def construct(self, *args, **kwargs):
        self.args.append(args)
        self.kwargs.append(kwargs)
        raise GeneratorExit(f"already catch first layer inputs, do not need continue.")


class OmniQuant(CompAlgo):
    """smooth quant for ptq"""

    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init OmniQuant with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig().inner_config(self._config, approach=PTQApproach.OMNI_QUANT)
        mode = self._config.mode
        self._is_deploy = mode == PTQMode.DEPLOY

        self.is_search_param = isinstance(self._config.algo_args.get("pre_clip_ratio"), list)

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, ds=None, **kwargs) -> Cell:
        """Apply"""
        if not network_helper:
            raise ValueError("Please provide network_helper when omni quant in apply phase.")

        if self._is_deploy:
            layers = network_helper.get_decoder_layers(network)
            for i in range(len(layers)):
                _, layer = layers[i]
                _, _, linears = network_helper.get_linears(layer)
                sq_linear_deploy_creator = partial(SQLinearDeploy, cfg=self._config)
                network_replace(layer, type(linears[0]), SQLinearDeploy, sq_linear_deploy_creator,
                                self._config.opname_blacklist)
            network.update_parameters_name()
            return network

        if not ds:
            raise ValueError("please provide dataset when use omni quant to quantize network.")
        catcher, network = self._get_first_layer_input(network, network_helper, ds, **kwargs)
        all_args = catcher.args
        all_kwargs = catcher.kwargs
        layers = network_helper.get_decoder_layers(network)
        for i in tqdm.tqdm(range(len(layers)), desc="Running OmniQuant..."):
            _, layer = layers[i]
            _, _, linears = network_helper.get_linears(layer)
            linear_type = type(linears[0])
            sq_linear_wrapper_creator = partial(SQLinearWrapper, cfg=self._config)
            network_replace(layer, linear_type, SQLinearWrapper, sq_linear_wrapper_creator,
                            self._config.opname_blacklist)
            _, _, linears = network_helper.get_linears(layer)

            layer.add_flags_recursive(infer_mode="observer_x")
            f_output = []
            for j in range(len(all_args)):
                cur_args = all_args[j]
                cur_kwargs = all_kwargs[j]
                f_output.append(layer(*cur_args, **cur_kwargs))

            if self._config.algo_args.get("is_revert_by_loss"):
                pass

            if not self.is_search_param:
                self._quantizer_linears(linears)
                for j in range(len(f_output)):
                    all_args[j] = list(all_args[j])
                    all_args[j][0] = f_output[j]
            else:
                all_args, all_kwargs = self._search_hyper_param(layer, linears,
                                                                f_output, all_args,
                                                                all_kwargs)
        network.update_parameters_name()
        if self.is_search_param:
            for _, ds_item in enumerate(ds.create_dict_iterator()):
                input_ids = ds_item['input_ids'].asnumpy()
                network_helper.generate(network, input_ids, max_new_tokens=1)
                break
        return network

    def _search_hyper_param(self, layer, linears, f_output, all_args, all_kwargs):
        """search hyper parameter"""
        layer.add_flags_recursive(infer_mode="quant")
        best_err = 1
        best_smooth_alpha = None
        best_pre_clip = None
        best_post_clip = None
        best_output = None
        for smooth_alpha in self._config.algo_args.get("smooth_alpha"):
            for pre_clip in self._config.algo_args.get("pre_clip_ratio"):
                for post_clip in self._config.algo_args.get("post_clip_ratio"):
                    for linear in linears:
                        linear.set_search_args(pre_clip_ratio=pre_clip, post_clip_ratio=post_clip,
                                               smooth_alpha=smooth_alpha)
                    q_output = []
                    for j in range(len(all_args)):
                        cur_args = all_args[j]
                        cur_kwargs = all_kwargs[j]
                        q_output.append(layer(*cur_args, **cur_kwargs))
                    loss_fn = MSELoss()
                    err = loss_fn(msops.cat(tuple(q_output)).astype(ms.float64),
                                  msops.cat(tuple(f_output)).astype(ms.float64)).asnumpy()
                    if err < best_err:
                        best_err = err
                        best_smooth_alpha = smooth_alpha
                        best_pre_clip = pre_clip
                        best_post_clip = post_clip
                        best_output = q_output
        for linear in linears:
            linear.set_search_args(pre_clip_ratio=best_pre_clip, post_clip_ratio=best_post_clip,
                                   smooth_alpha=best_smooth_alpha)
        for j in range(len(best_output)):
            all_args[j] = list(all_args[j])
            all_args[j][0] = best_output[j]
        return all_args, all_kwargs

    def _quantizer_linears(self, linears):
        for linear in linears:
            if isinstance(linear, SQLinearWrapper):
                linear.set_search_args(pre_clip_ratio=self._config.algo_args.get("pre_clip_ratio"),
                                       post_clip_ratio=self._config.algo_args.get("post_clip_ratio"),
                                       smooth_alpha=self._config.algo_args.get("smooth_alpha"))
                linear.quant_weight()

    def _get_first_layer_input(self, network: Cell, network_helper: NetworkHelper = None, ds=None, **kwargs):
        """get first layer input"""
        layers = network_helper.get_decoder_layers(network)
        catcher = InputCatcher(layers[0][1])

        def replace_first_decoder(root: Cell, src: Cell, dst: Cell):
            if root is None:
                return
            for name, cell in root.name_cells().items():
                if cell is src:
                    root.insert_child_to_cell(name, dst)
                    return
                replace_first_decoder(cell, src, dst)

        replace_first_decoder(network, layers[0][1], catcher)
        if not ds:
            raise ValueError("OmniQuant need dataset to calibrate, please provide dataset.")
        ds_count = kwargs.get("ds_count", None)
        if ds_count:
            ds_count = int(ds_count)
        total_count = ds.get_dataset_size()
        total_count = ds_count if ds_count and ds_count < total_count else total_count
        data_count = 1
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            try:
                network_helper.generate(network, input_ids, max_new_tokens=1)
            except GeneratorExit:
                if network.block_mgr:
                    network.block_mgr.clear_cache()
                if data_count >= total_count:
                    break
                data_count += 1
                continue
        replace_first_decoder(network, catcher, catcher.handler)
        offload_network(network)
        return catcher, network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """convert"""
        if not isinstance(net_opt, Cell):
            raise TypeError(
                f'The parameter `net_opt` must be isinstance of Cell, but got {type(net_opt)}.')
        if not isinstance(ckpt_path, str):
            raise TypeError(
                f'The parameter `ckpt_path` must be isinstance of str, but got {type(ckpt_path)}.')
        real_path = os.path.realpath(ckpt_path)
        if ckpt_path != "":
            if os.path.isfile(real_path):
                param_dict = load_checkpoint(ckpt_path)
                load_param_into_net(net_opt, param_dict)
            else:
                raise ValueError(
                    f'The parameter `ckpt_path` can only be empty or a valid file, but got {real_path}.')
        return net_opt
