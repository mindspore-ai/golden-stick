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
"""PTQ algorithm."""
from typing import List
import time
import os
import copy
import tqdm
from mindspore import dtype
from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_network
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQApproach, PTQMode
from mindspore_gs.ptq.network_helpers import NetworkHelper
from .algorithm import Algorithm
from .algorithms import LinearSmoother, Quantizer, Deployer


class InputCatcher(Cell):
    """input catcher"""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.attention = handler.attention
        self.args = []
        self.kwargs = []

    def construct(self, *args, **kwargs):
        self.args.append(list(args))
        self.kwargs.append(kwargs)
        raise GeneratorExit(f"already catch first layer inputs, do not need continue.")


class PTQ(CompAlgo):
    """ptq"""

    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init PTQ with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig().inner_config(self._config, approach=PTQApproach.PTQ)
        logger.info(f"Config for PTQ: {self._config}")
        self.pipeline: List[Algorithm] = []
        self.build_pipeline()

    def build_pipeline(self):
        """build pipline"""
        if self._config.mode == PTQMode.QUANTIZE:
            if self._config.outliers_suppression == 'smooth':
                logger.info("Adding LinearSmoother to pipeline.")
                LinearSmoother.load_mindformers_plugin()
                self.pipeline.append(LinearSmoother(self._config))
            if self._config.act_quant_dtype == dtype.int8 or \
                self._config.weight_quant_dtype == dtype.int8 or \
                    self._config.kvcache_quant_dtype == dtype.int8:
                logger.info("Adding Quantizer to pipeline.")
                Quantizer.load_mindformers_plugin()
                self.pipeline.append(Quantizer(self._config))
        elif self._config.mode == PTQMode.DEPLOY:
            logger.info("Adding Deploy to pipeline.")
            Deployer.load_mindformers_plugin()
            self.pipeline.append(Deployer(self._config))

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, ds=None, **kwargs) -> Cell:
        """Apply"""
        if self._config.mode == PTQMode.DEPLOY:
            layers = network_helper.get_decoder_layers(network)
            for i in tqdm.tqdm(range(len(layers)), desc="Running PTQ Deploy..."):
                layer_name, layer = layers[i]
                for processor in self.pipeline:
                    processor.process(layer_name, layer, None, None, network_helper)
                    network.update_parameters_name()
            return network
        if not network_helper:
            raise ValueError("Please provide network_helper when omni quant in apply phase.")
        if not ds:
            raise ValueError("please provide dataset when use omni quant to quantize network.")
        start_time = time.time()
        logger.info("Analysis network structure.")
        network_helper.analysis_decoder_groups(network)
        logger.info(f"analysis_decoder_groups time cost {time.time() - start_time}")
        start_time = time.time()
        logger.info(f"Catching inputs for first decoder layer with {ds.get_dataset_size()} samples from datasets.")
        catcher, network = PTQ._get_first_layer_input(network, network_helper, ds)
        all_args = catcher.args
        all_kwargs = catcher.kwargs
        logger.info(f"_get_first_layer_input time cost {time.time() - start_time}")
        start_time = time.time()
        layers = network_helper.get_decoder_layers(network)
        logger.info(f"get_decoder_layers time cost {time.time() - start_time}")
        for i in tqdm.tqdm(range(len(layers)), desc="Running PTQ..."):
            logger.info(f"Quantize {i}th decoder layer.")
            layer_name, layer = layers[i]
            start_time = time.time()
            cur_args, cur_kwargs = copy.deepcopy(all_args), copy.deepcopy(all_kwargs)
            for args, kwargs in zip(all_args, all_kwargs):
                args[0] = layer(*args, **kwargs)
            logger.info(f"{i}th layer get net next layer input time cost {time.time() - start_time}")
            for processor in self.pipeline:
                start_time = time.time()
                processor.process(layer_name, layer, cur_args, cur_kwargs, network_helper)
                processor.deploy(layer_name, layer)
                network.update_parameters_name()
                logger.info(f"{i}th layer do {type(processor)} time cost {time.time() - start_time}")
            start_time = time.time()
            offload_network(layer)
            logger.info(f"{i}th layer offload network time cost {time.time() - start_time}")
        return network

    @staticmethod
    def _get_first_layer_input(network: Cell, network_helper: NetworkHelper = None, ds=None):
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
        total_count = ds.get_dataset_size()
        data_count = 1
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            try:
                network_helper.generate(network, input_ids, max_new_tokens=1)
            except GeneratorExit:
                if network.block_mgr:
                    network.block_mgr.clear_cache()
            data_count += 1
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
