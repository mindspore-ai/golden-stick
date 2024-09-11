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
from typing import List, Union, Tuple
import time
import os
import copy
import tqdm
from mindspore import dtype, get_context, PYNATIVE_MODE
from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_network, value_check
from mindspore_gs.ptq.processor import transform_network_inplace
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQApproach, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.processor import Processor
from .algorithm import Algorithm
from .algorithms import LinearSmoother, Quantizer, Deployer


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
        self.args.append(list(args))
        self.kwargs.append(kwargs)
        raise GeneratorExit("already catch first layer inputs, do not need continue.")


class PTQ(CompAlgo):
    """
    Implementation of PTQ algorithm which supports the combination quantization of activation,
    weight, and kvcache.

    Args:
        config(:class:`mindspore_gs.ptq.PTQConfig`): config for PTQ, default is ``None``.

    Raises:
        TypeError: If `config` type is not PTQConfig when it's not ``None``.
        ValueError: If not PYNATIVE mode when mode in config is PTQMode.QUANTIZE.
        ValueError: If act_quant_dtype is int8 and weight_quant_dtype is None.

    Examples:
        >>> import mindspore_gs
        >>> from mindspore_gs.ptq import PTQ
        >>> from mindspore_gs.ptq import PTQConfig
        >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
        >>> from mindformers.tools.register.config import MindFormerConfig
        >>> from mindformers import LlamaForCausalLM, LlamaConfig
        >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
        >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
        >>> helper = MFLlama2Helper(mfconfig)
        >>> ptq_config = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
        >>> ptq = PTQ(ptq_config)
        >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
        >>> fake_quant_net = ptq.apply(network, helper)
        >>> quant_net = ptq.convert(fake_quant_net)
    """

    def __init__(self, config: Union[dict, PTQConfig] = None):
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
        PTQ._ptq_config_check(self._config)
        self.pipeline: List[Algorithm] = []
        self.decoder_layer_types: list = []
        self.build_pipeline()
        self._load_mindformers_plugin()
        self.context_mode = get_context("mode")

    def build_pipeline(self):
        """build pipline"""
        if self._config.mode == PTQMode.QUANTIZE:
            if self._config.outliers_suppression == OutliersSuppressionType.SMOOTH:
                logger.info("Adding LinearSmoother to pipeline.")
                self.pipeline.append(LinearSmoother(self._config))
            if self._config.act_quant_dtype == dtype.int8 or \
                self._config.weight_quant_dtype == dtype.int8 or \
                    self._config.kvcache_quant_dtype == dtype.int8:
                logger.info("Adding Quantizer to pipeline.")
                self.pipeline.append(Quantizer(self._config))
        elif self._config.mode == PTQMode.DEPLOY:
            logger.info("Adding Deploy to pipeline.")
            self.pipeline.append(Deployer(self._config))

    def _load_mindformers_plugin(self):
        for algorithm in self.pipeline:
            algorithm.load_mindformers_plugin()
        from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
        from mindformers.experimental.infer.core.transformer import ParallelTransformerLayer
        self.decoder_layer_types.append(LLamaDecodeLayer)
        self.decoder_layer_types.append(ParallelTransformerLayer)

    def _get_decoder_layers(self, network: Cell):
        """
        Get decoder layers from network.

        Args:
            network (nn.Cell): Network to get decoder layers.

        Returns:
            A list of tuples (cell_name, `Cell`) as decoder layers of network.
        """
        value_check('network', network, Cell)

        class NetworkWalker(Processor):
            def __init__(self, decoder_layer_types_):
                self.layers = []
                self._decoder_layer_types = decoder_layer_types_

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if isinstance(cell, self._decoder_layer_types):
                    self.layers.append((cell_name, cell))
                    return cell, True
                return cell, False

        walker = NetworkWalker(tuple(self.decoder_layer_types))
        walker.process(network)
        return walker.layers

    @staticmethod
    def _ptq_config_check(config):
        """_ptq_config_check"""
        if config.outliers_suppression is None and \
            config.weight_quant_dtype == dtype.int8 and \
                config.act_quant_dtype == dtype.int8:
            logger.warning("When outliers_suppression is None, A8W8 algorithm accuracy is expected to decline.")
        if config.weight_quant_dtype is None and \
                config.act_quant_dtype == dtype.int8:
            raise ValueError("PTQ algorithm not support only quant activation.")
        if config.weight_quant_dtype is None and config.act_quant_dtype is None \
            and config.kvcache_quant_dtype is None and \
                config.outliers_suppression == OutliersSuppressionType.NONE:
            logger.warning("PTQ algorithm does not quantify any layers when"
                           "weight_quant_dtype=None,"
                           "act_quant_dtype=None,"
                           "kvcache_quant_dtype=None and"
                           "outliers_suppression=None")

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, datasets=None, **kwargs) -> Cell:
        """
        Define how to add fake quantizer to `network`.

        Args:
            network (Cell): Network to be fake quantized.
            network_helper (NetworkHelper): Utils for decoupling algorithm with network framework.
            datasets (Dataset): Datasets for calibrating.

        Returns:
            fake quantized network.

        Raises:
            RuntimeError: If PTQ is not well inited.
            TypeError: If input `network` is not a Cell.
            ValueError: If input `network_helper` is None when mode is `PTQMode.DEPLOY`.
            ValueError: If input datasets is None.
        """
        if self._config.mode == PTQMode.DEPLOY:
            layers = self._get_decoder_layers(network)
            for i in tqdm.tqdm(range(len(layers)), desc="Running PTQ Deploy..."):
                layer_name, layer = layers[i]
                for processor in self.pipeline:
                    processor.replace(layer_name, layer)
                    processor.process(layer_name, layer)
                    processor.deploy(layer_name, layer)
                    network.update_parameters_name()
            return network
        if self._config.mode == PTQMode.QUANTIZE and get_context("mode") != PYNATIVE_MODE:
            raise ValueError("Quantization phase only support PYNATIVE MODE.")
        if not network_helper:
            raise ValueError("Please provide network_helper when PTQ in apply phase.")
        if not datasets:
            raise ValueError("please provide dataset when use PTQ quant to quantize network.")
        if self._config.kvcache_quant_dtype == dtype.int8 and not network_helper.get_spec("use_past"):
            raise ValueError("use_past need be true when doing kvcache quantize.")
        logger.info(f"Visible decoder layer types: {self.decoder_layer_types}. If decoder layer type of target network "
                    "not in list, please modify PTQ.decoder_layer_types before invoking apply method.")
        start_time = time.time()
        logger.info("Analysis network structure.")
        network_helper.analysis_decoder_groups(network)
        logger.info(f"analysis_decoder_groups time cost {time.time() - start_time}")
        start_time = time.time()
        logger.info(f"Catching inputs for first decoder layer with {datasets.get_dataset_size()} datasets samples.")
        catcher, network = self._get_first_layer_input(network, network_helper, datasets)
        all_args = catcher.args
        all_kwargs = catcher.kwargs
        logger.info(f"_get_first_layer_input time cost {time.time() - start_time}")
        start_time = time.time()
        layers = self._get_decoder_layers(network)
        if not layers:
            logger.warning(
                f"No decoder layer found in network. Visible decoder layer types: {self.decoder_layer_types}, "
                "please modify PTQ.decoder_layer_types before invoking apply method.")
        else:
            logger.info(f"get_decoder_layers time cost {time.time() - start_time}")
        for i in tqdm.tqdm(range(len(layers)), desc="Running PTQ..."):
            logger.info(f"Quantize {i}th decoder layer.")
            layer_name, layer = layers[i]
            cur_args, cur_kwargs = copy.deepcopy(all_args), copy.deepcopy(all_kwargs)
            for processor in self.pipeline:
                processor.replace(layer_name, layer, network_helper)

                logger.info("Catching inputs of all Linear in decoder layer.")
                start_time = time.time()

                transform_network_inplace(layer, WrapperCell, lambda _, cell: cell.add_hook())
                index = 0
                for args, kwargs in zip(cur_args, cur_kwargs):
                    all_args[index][0] = layer(*args, **kwargs)
                    index += 1

                transform_network_inplace(layer, WrapperCell, lambda _, cell: cell.remove_hook())
                logger.info(f"{i}th layer output refresh time cost {time.time() - start_time}")

                start_time = time.time()
                processor.process(layer_name, layer, network_helper)
                processor.deploy(layer_name, layer)
                network.update_parameters_name()
                logger.info(f"{i}th layer do {type(processor)} time cost {time.time() - start_time}")

            start_time = time.time()
            offload_network(layer)
            logger.info(f"{i}th layer offload network time cost {time.time() - start_time}")
        return network

    def _get_first_layer_input(self, network: Cell, network_helper: NetworkHelper = None, ds=None):
        """get first layer input"""
        layers = self._get_decoder_layers(network)
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
            raise ValueError("PTQ need dataset to calibrate, please provide dataset.")
        total_count = ds.get_dataset_size()
        data_count = 1
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            try:
                network_helper.generate(network, input_ids, max_new_tokens=1)
            except GeneratorExit:
                if hasattr(network, "block_mgr"):
                    network.block_mgr.clear_cache()
            data_count += 1
        replace_first_decoder(network, catcher, catcher.handler)
        offload_network(network)
        return catcher, network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `RoundToNearest.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is ``""``, which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents quantized network.

        Raises:
            TypeError: If `net_opt` is not Cell.
            TypeError: If `ckpt_path` is not string.
            ValueError: If `ckpt_path` is not empty and invalid.
        """
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
