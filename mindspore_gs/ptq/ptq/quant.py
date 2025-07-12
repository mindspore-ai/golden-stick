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
from functools import partial
from typing import List, Union, Tuple, Optional
from collections import OrderedDict
import time
import gc
import os
import copy
import tqdm
from mindspore import dtype, get_context, PYNATIVE_MODE
from mindspore.nn import Cell
from mindspore.nn.utils import no_init_parameters
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_network, value_check
from mindspore_gs.ptq.processor import transform_network_inplace
from mindspore_gs.ptq.ptq_config import PTQConfig, PTQMode, OutliersSuppressionType, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig, PTQApproach
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell, SearchInputs
from mindspore_gs.ptq.processor import Processor
from .algorithm import Algorithm
from .algorithms import LinearSmoothQuant, LinearAutoSmoother, LinearClipper, Quantizer


class InputCatcher(Cell):
    """input catcher"""

    def __init__(self):
        super().__init__()
        self.handler = None
        self.args = []
        self.kwargs = []
        self.old_construct = None
        self.patched = False

    def patch(self, handler):
        """patch"""
        if self.patched:
            raise RuntimeError("Only support patch one cell for one time. please invoke recover before invoking patch "
                               "again.")
        self.handler = handler
        self.old_construct = handler.construct
        self.handler.construct = partial(InputCatcher.construct, self)
        self.patched = True

    def recover(self):
        """recover"""
        if self.patched and self.handler and self.old_construct:
            self.handler.construct = self.old_construct
        self.patched = False

    def construct(self, *args, **kwargs):
        """construct"""
        self.args.append(list(args))
        self.kwargs.append(kwargs)
        raise GeneratorExit("already catch first layer inputs, do not need continue.")


class PTQ(CompAlgo):
    """
    Implementation of PTQ algorithm which supports the combination quantization of activation,
    weight, and kvcache.

    Args:
        config(:class:`mindspore_gs.ptq.PTQConfig`, optional): config for PTQ, default is ``None``.
        layer_policies(OrderedDict, optional): quantization strategy for layers, default is ``None``.
            The key of `layer_policies` is regular string to match the layer name,
            the value of `layer_policies` is :class:`mindspore_gs.ptq.PTQConfig`.

    Raises:
        TypeError: If `config` type is not PTQConfig when it's not ``None``.
        TypeError: If any value in `layer_policies` type is not PTQConfig when it's not ``None``.
        ValueError: If not PYNATIVE mode when mode in config is PTQMode.QUANTIZE.
        ValueError: If act_quant_dtype is int8 and weight_quant_dtype is None.
        TypeError: If layer_policies is not an OrderedDict.

    Examples:
        >>> import mindspore_gs
        >>> from mindspore_gs.ptq import PTQ
        >>> from mindspore_gs.ptq import PTQConfig
        >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
        >>> from mindformers.tools.register.config import MindFormerConfig
        >>> from mindformers import LlamaForCausalLM, LlamaConfig
        >>> from mindspore_gs.common.gs_enum import BackendTarget
        >>> from mindspore import dtype as msdtype
        >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
        >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
        >>> helper = MFLlama2Helper(mfconfig)
        >>> backend = BackendTarget.ASCEND
        >>> ptq_config = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"],
        ...                        weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
        ...                        outliers_suppression=OutliersSuppressionType.SMOOTH)
        >>> attn_policy = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend,
        ...                         weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
        ...                         outliers_suppression=OutliersSuppressionType.NONE)
        >>> layer_policy = OrderedDict({r'.*Attention.wo.*': attn_policy})
        >>> ptq = PTQ(ptq_config, layer_policy)
        >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
        >>> fake_quant_net = ptq.apply(network, helper)
        >>> quant_net = ptq.convert(fake_quant_net)
        >>> ptq.summary(quant_net)
    """

    def __init__(self, config: Union[dict, PTQConfig] = None, layer_policies=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init PTQ with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        if layer_policies is None:
            self.layer_policies = OrderedDict()
        else:
            self.layer_policies = layer_policies
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig().inner_config(self._config, approach=PTQApproach.PTQ)
        self._generate_func = None
        logger.info(f"Config for PTQ: {self._config}")
        PTQ._ptq_config_check(self._config)
        self._layer_policies_check()
        self.pipeline: List[Algorithm] = []
        self.decoder_layers: list[Cell] = []
        self.decoder_layer_types: list = []
        self.context_mode = get_context("mode")
        self._target_layer_type = ()
        self._build_pipeline()
        self._load_mindformers_plugin()

    def _append_algorithm(self, name, algorithm: Algorithm):
        logger.info(f"append {name} to pipeline.")
        self.pipeline.append(algorithm)

    def _build_pipeline(self):
        """build pipline"""
        smoothquant = LinearSmoothQuant(self._config, self.layer_policies)
        clipper = LinearClipper(self._config, self.layer_policies)
        awq = LinearAutoSmoother(self._config, self.layer_policies)
        quantizer = Quantizer(self._config, self.layer_policies)
        self._append_algorithm('LinearSmoothQuant', smoothquant)
        self._append_algorithm('LinearAutoSmoother', awq)
        self._append_algorithm('LinearClipper', clipper)
        self._append_algorithm('Quantizer', quantizer)

    def _load_mindformers_plugin(self):
        """_load_mindformers_plugin"""
        for algorithm in self.pipeline:
            algorithm.load_mindformers_plugin()
            self._target_layer_type += algorithm.target_layer_type()
        from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
        self.decoder_layer_types.append(LLamaDecodeLayer)
        try:
            from mindformers.experimental.infer.core.transformer import ParallelTransformerLayer
            self.decoder_layer_types.append(ParallelTransformerLayer)
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.transformer import ParallelTransformerLayer as LlamaParallelTransformerLayer
            from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
            self.decoder_layer_types.append(DeepseekV3DecodeLayer)
            self.decoder_layer_types.append(LlamaParallelTransformerLayer)
        except ImportError:
            pass
        try:
            from research.telechat2.infer.telechat_transformers import TelechatParallelTransformerLayer
            self.decoder_layer_types.append(TelechatParallelTransformerLayer)
        except ImportError:
            pass

        def generate(network, input_ids, helper=None):
            if isinstance(helper, NetworkHelper):
                return helper.generate(network, input_ids, do_sample=False, max_new_tokens=1)
            return network.generate(input_ids, do_sample=False, max_new_tokens=1)
        self._generate_func = generate

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
        if walker.layers:
            self.decoder_layers = walker.layers
            return
        self.decoder_layers = [("network", network)]
        logger.warning(
            f"No decoder layer found in network. Visible decoder layer types: {self.decoder_layer_types}, "
            "please modify PTQ.decoder_layer_types before invoking apply method. If not, PTQ will take lots of memory.")

    @staticmethod
    def _ptq_config_check(config):
        """_ptq_config_check"""
        use_w8 = config.weight_quant_dtype == dtype.int8
        use_a8 = config.act_quant_dtype == dtype.int8
        if config.outliers_suppression is None and use_a8 and use_w8:
            logger.warning("When outliers_suppression is None, A8W8 algorithm accuracy is expected to decline.")
        if config.weight_quant_dtype is None and use_a8:
            raise ValueError("PTQ algorithm do not support only quant activation.")

        use_ptq_or_awq = (config.outliers_suppression == OutliersSuppressionType.AWQ or
                          config.precision_recovery == PrecisionRecovery.GPTQ)
        if use_w8 and use_a8 and use_ptq_or_awq:
            raise ValueError("AWQ algorithm and GPTQ algorithm do not support quant activation.")

        use_a8w8_only = use_a8 and use_w8 and config.kvcache_quant_dtype is None
        use_osl = config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE
        if not use_a8w8_only and use_osl:
            raise ValueError("OUTLIER_SUPPRESSION_LITE algorithm only support W8A8 quant.")

        use_w4 = config.weight_quant_dtype == dtype.qint4x2
        use_c8 = config.kvcache_quant_dtype == dtype.int8
        if use_w4 and use_c8:
            raise ValueError("PTQ algorithm only support quant weight in int4 alone."
                             "Please not to use with c8 at the same time.")

    def _layer_policies_check(self):
        """_layer_policies_check"""
        import re
        if not isinstance(self.layer_policies, OrderedDict):
            raise TypeError(f'layer_policies should be an OrderedDict, bug got {type(self.layer_policies)}.')
        if any(not isinstance(key, str) for key in self.layer_policies.keys()):
            raise TypeError(f'all key of layer_policies should be a string.')
        try:
            for key, config_ in self.layer_policies.items():
                if config_:
                    re.compile(key)
                    if not isinstance(config_, PTQConfig):
                        raise TypeError(f'The type of value in layer_policies should be PTQConfig,'
                                        f'but got {type(config_)}')
                    if config_.mode != self._config.mode:
                        logger.warning(f'The mode={config_.mode} in {key} layer policy different from '
                                       f'mode={self._config.mode} in network policy, PTQ algorithm use network policy '
                                       f'mode to quant.')
                        config_.mode = self._config.mode
                    if config_.backend != self._config.backend:
                        logger.warning(f'The backend={config_.backend} in {key} layer policy different from '
                                       f'backend={self._config.backend} in network policy, PTQ algorithm use network '
                                       f'policy backend to quant.')
                        config_.backend = self._config.backend
                    self.layer_policies[key] = InnerPTQConfig().inner_config(config_, approach=PTQApproach.PTQ)
                    PTQ._ptq_config_check(self.layer_policies[key])
        except re.error:
            raise TypeError('The regular string of layer_policies not correct, please check and try again.') \
                from re.error

    # pylint: disable=arguments-differ
    # pylint: disable=unused-argument
    def apply(self, network: Cell,
              network_helper: NetworkHelper = None,
              datasets=None, **kwargs) -> Cell:
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
            ValueError: If input datasets is None.
        """
        self._config.update_comm_info()
        self._get_decoder_layers(network)
        if self._config.mode == PTQMode.DEPLOY:
            logger.info("unset environ FORCE_EAGER and MS_JIT because of PTQMode.DEPLOY mode")
            for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running PTQ Deploy..."):
                layer_name, layer = self.decoder_layers[i]
                for processor in self.pipeline:
                    with no_init_parameters():
                        processor.replace(layer_name, layer)
                        processor.deploy(layer_name, layer)
                    network.update_parameters_name()
            return network
        os.environ['FORCE_EAGER'] = 'true'
        logger.info("set environ FORCE_EAGER=true and MS_JIT=0 because of PTQMode.QUANTIZE mode")
        if get_context("mode") != PYNATIVE_MODE:
            raise ValueError("In QUANTIZE phase, please set mode=PYNATIVE_MODE.")
        if not datasets:
            raise ValueError("please provide dataset when use PTQ quant to quantize network.")
        logger.info(f"Visible decoder layer types: {self.decoder_layer_types}. If decoder layer type of target network "
                    "not in list, please modify PTQ.decoder_layer_types before invoking apply method.")
        logger.info("Analysis network structure.")
        start_time = time.time()
        logger.info(f"Catching inputs for first decoder layer with {datasets.get_dataset_size()} datasets samples.")
        catcher, network = self._get_first_layer_input(network, datasets, network_helper)
        all_args = catcher.args
        all_kwargs = catcher.kwargs
        logger.info(f"_get_first_layer_input time cost {time.time() - start_time}")
        start_time = time.time()
        logger.info(f"get_decoder_layers time cost {time.time() - start_time}")
        for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running PTQ..."):
            logger.info(f"Quantize {i}th decoder layer.")
            layer_name, layer = self.decoder_layers[i]
            cur_args, cur_kwargs = copy.deepcopy(all_args), copy.deepcopy(all_kwargs)
            if self._config.always_use_fp_input_in_processer:
                for index, (args, kwargs) in enumerate(zip(cur_args, cur_kwargs)):
                    output = layer(*args, **kwargs)
                    if len(self.decoder_layers) > 1:
                        all_args[index][0] = output[0] if isinstance(output, tuple) else output
            for processor in self.pipeline:
                processor.replace(layer_name, layer, search_inputs=SearchInputs(layer, cur_args, cur_kwargs))

                logger.info("Catching inputs of all Linear in decoder layer.")
                start_time = time.time()

                transform_network_inplace(layer, WrapperCell, lambda _, cell: cell.add_hook())
                index = 0
                for args, kwargs in zip(cur_args, cur_kwargs):
                    output = layer(*args, **kwargs)
                    if len(self.decoder_layers) > 1 and not self._config.always_use_fp_input_in_processer:
                        # FIXME: 'always_use_fp_input_in_processer' is a temporary switch for fixing activation between
                        # layers. This branch may introduces error to the next layer, because previous processors in the
                        # pipeline changes the layer, and thus, gives a inaccurate output. Set the switch to True to
                        # avoid this issue. The switch should be removed after the issue is fixed. -- @tongl2
                        all_args[index][0] = output[0] if isinstance(output, tuple) else output
                    index += 1

                transform_network_inplace(layer, WrapperCell, lambda _, cell: cell.remove_hook())
                logger.info(f"{i}th layer output refresh time cost {time.time() - start_time}")

                processor.process(layer_name, layer)
                processor.deploy(layer_name, layer)
                network.update_parameters_name()
                gc.collect()
            if self._config.reflash_inputs_after_each_processor:
                index = 0
                for args, kwargs in zip(cur_args, cur_kwargs):
                    all_args[index][0] = layer(*args, **kwargs)
                    index += 1
            start_time = time.time()
            offload_network(layer)
            gc.collect()
            logger.info(f"{i}th layer offload network time cost {time.time() - start_time}")
        return network

    def _get_first_layer_input(self, network: Cell, ds=None, helper=None):
        """get first layer input"""
        catcher = InputCatcher()
        catcher.patch(self.decoder_layers[0][1])
        if not ds:
            raise ValueError("PTQ need dataset to calibrate, please provide dataset.")
        total_count = ds.get_dataset_size()
        data_count = 1
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            try:
                self._generate_func(network, input_ids, helper)
            except GeneratorExit:
                if hasattr(network, "block_mgr") and network.block_mgr:
                    network.block_mgr.clear_cache()
            data_count += 1
        catcher.recover()
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
        self.summary(net_opt)
        return net_opt

    def _summary_target_layer_type(self) -> tuple:
        return self._target_layer_type

    def _summary_layer(self, layer_name, layer: Cell) -> Optional[str]:
        info = self._config.layer_quant_info_collect.get(layer_name)
        if not info and layer_name.endswith('_layer'):
            info = self._config.layer_quant_info_collect.get(layer_name[:-7])
        if not info and layer_name.endswith('.layer'):
            info = self._config.layer_quant_info_collect.get(layer_name[:-6])
        return info

    def _summary_title(self):
        return "Network Quantization Summary"

    def _summary_desc_name(self):
        return "quant_type"
