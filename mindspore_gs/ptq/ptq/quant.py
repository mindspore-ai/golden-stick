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
from typing import List, Union, Tuple, Optional, Callable
from collections import OrderedDict
import time
import gc
import os
import copy
import tqdm
import numpy as np

from datasets  import Dataset
from mindspore import dtype, get_context, PYNATIVE_MODE
from mindspore.nn import Cell
from mindspore.nn.utils import no_init_parameters
from mindspore.dataset import GeneratorDataset, RepeatDataset
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_network, value_check
from mindspore_gs.ptq.basic_functions.processor import transform_network_inplace
from mindspore_gs.ptq.ptq_config import PTQConfig, PTQMode, OutliersSuppressionType, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig, PTQApproach
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell, SearchInputs
from mindspore_gs.ptq.basic_functions.processor import Processor
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


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

def convert_to_dataset(datasets):
    """Convert non-Dataset data to Dataset"""
    if isinstance(datasets, Dataset):
        return datasets
    if isinstance(datasets, (GeneratorDataset, RepeatDataset)):
        samples = [sample['input_ids'].asnumpy() \
            for sample in datasets.create_dict_iterator()]
        datasets = Dataset.from_dict({"input_ids": samples})
        datasets.set_transform(lambda examples:
                               {"input_ids": np.array(examples["input_ids"])})
        return datasets
    raise TypeError(f"Unsupported data type: {type(datasets)}. "
                    "Please provide a Dataset, GeneratorDataset or RepeatDataset.")

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

    pipeline: List = []
    target_layer_type: tuple = ()

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
        self.decoder_layers: list[Cell] = []
        self.decoder_layer_types: list = []
        self.context_mode = get_context("mode")

    def _load_mindformers_plugin(self):
        """_load_mindformers_plugin"""
        try:
            from mindspore_gs.ptq.plugins import MFModelHubPlugin
            # pylint: disable=protected-access
            MFModelHubPlugin()._load_quant_cells()
            MFModelHubPlugin()._load_algo_modules()
        except ImportError:
            pass
        try:
            from mindformers.models.llama.llama_transformer import LLamaDecodeLayer
            self.decoder_layer_types.append(LLamaDecodeLayer)
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

    def set_ptq_config(self, **kwargs):
        """Set PTQ config.
        
        .. note::
            This is an internal API method, not intended for external use.
            It is used internally by model implementations (e.g., MindOneModel, MFModel)
            to configure PTQ settings. External users should use the PTQConfig
            passed to the PTQ constructor instead.

        Args:
            **kwargs: Configuration parameters to set. Each key should be a valid
                attribute of PTQConfig.
        
        Raises:
            AttributeError: If any key in kwargs is not a valid PTQConfig attribute.
            TypeError: If the type of any value doesn't match the expected type.
        """
        for key, value in kwargs.items():
            if not hasattr(self._config, key):
                raise AttributeError(f"'{type(self._config).__name__}' "
                                     f"object has no attribute '{key}', "
                                     "please check and try again.")
            if not isinstance(value, type(getattr(self._config, key))):
                raise TypeError(f"The type of value for '{key}' in PTQConfig "
                                f"should be {type(getattr(self._config, key))}, "
                                f"but got {type(value)}")
            setattr(self._config, key, value)

    def set_generate_func(self, generate_func: Callable):
        """Set generate function for getting first layer input.
        
        .. note::
            This is an internal API method, not intended for external use.
            It is used internally by model implementations (e.g., MFModel)
            to set the generate function for capturing first layer inputs during calibration.
            External users should not need to call this method directly.
        
        Args:
            generate_func (Callable): The function to use for generating inputs.
                This function should accept dataset items and return model outputs.
        """
        self._generate_func = generate_func

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
            raise TypeError('all key of layer_policies should be a string.')
        try:
            for key, config_ in self.layer_policies.items():
                if config_:
                    re.compile(key)
                    if not isinstance(config_, PTQConfig):
                        raise TypeError('The type of value in layer_policies should be PTQConfig,'
                                        f'but got {type(config_)}')
                    if config_.mode != self._config.mode:
                        logger.warning(f'The mode={config_.mode} in {key} layer policy different from '
                                       f'mode={self._config.mode} in network policy, PTQ algorithm use network policy '
                                       'mode to quant.')
                        config_.mode = self._config.mode
                    if config_.backend != self._config.backend:
                        logger.warning(f'The backend={config_.backend} in {key} layer policy different from '
                                       f'backend={self._config.backend} in network policy, PTQ algorithm use network '
                                       'policy backend to quant.')
                        config_.backend = self._config.backend
                    self.layer_policies[key] = InnerPTQConfig().inner_config(config_, approach=PTQApproach.PTQ)
                    PTQ._ptq_config_check(self.layer_policies[key])
        except re.error:
            raise TypeError('The regular string of layer_policies not correct, please check and try again.') \
                from re.error

    def _check_apply_inputs(self, datasets):
        """_check_apply_inputs"""
        os.environ['ENFORCE_EAGER'] = 'true'
        logger.info("set environ ENFORCE_EAGER=true and MS_JIT=0 because of PTQMode.QUANTIZE mode")
        if get_context("mode") != PYNATIVE_MODE:
            raise ValueError("In QUANTIZE phase, please set mode=PYNATIVE_MODE.")
        if not datasets:
            raise ValueError("please provide dataset when use PTQ quant to quantize network.")
        if not isinstance(datasets, Dataset):
            raise RuntimeError(f"The type of dataset is not correct, suppose to {Dataset.__class__.__name__}, "
                               f"but got {type(datasets)}")
        logger.info(f"Visible decoder layer types: {self.decoder_layer_types}. If decoder layer type of target network "
                    "not in list, please modify PTQ.decoder_layer_types before invoking apply method.")
        logger.info("Analysis network structure.")

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
        framework = kwargs.get('framework', "mindformers")
        if framework == "mindone":
            return self._apply_mindone(network, datasets, **kwargs)
        if framework == "mindformers":
            return self._apply_mindformers(network, network_helper, datasets, **kwargs)
        raise ValueError(f"Invalid framework: {framework}. Please use 'mindone' or 'mindformers'.")

    # pylint: disable=unused-argument
    def _apply_mindformers(self, network: Cell,
              network_helper: NetworkHelper = None,
              datasets=None, **kwargs) -> Cell:
        """
        Define how to add fake quantizer to `network` for mindformers framework.
        """
        def catch_layer_output(layer, input_args, input_kwargs, output_args, output_kwargs, do_update=True):
            for index, (args, kwargs) in enumerate(zip(input_args, input_kwargs)):
                output = layer(*args, **kwargs)
                if do_update:
                    if "hidden_states" in all_kwargs[index]:
                        output_kwargs[index]["hidden_states"] = output[0] if isinstance(output, tuple) else output
                    else:
                        output_args[index][0] = output[0] if isinstance(output, tuple) else output
        # FIXME: This is a temporary solution to load mindformers plugin when experimental is False.
        # This should be removed after the research network from mindformers is not supported by PTQ. -- @yyyyrf
        if not self._config.experimental:
            self._load_mindformers_plugin()

        self._config.update_comm_info()
        self._get_decoder_layers(network)
        if self._config.mode == PTQMode.DEPLOY:
            for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running PTQ Deploy..."):
                layer_name, layer = self.decoder_layers[i]
                for processor in PTQ.pipeline:
                    processor = processor(self._config, self.layer_policies)
                    with no_init_parameters():
                        processor.replace(layer_name, layer)
                        processor.deploy(layer_name, layer)
                    network.update_parameters_name()
            return network

        datasets = convert_to_dataset(datasets)
        self._check_apply_inputs(datasets)
        start_time = time.time()
        logger.info(f"Catching inputs for first decoder layer with {len(datasets)} datasets samples.")
        catcher, network = self._get_mf_first_layer_input(network, datasets, network_helper)
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
                catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs,
                                   do_update=len(self.decoder_layers) > 1)
            for processor in PTQ.pipeline:
                processor = processor(self._config, self.layer_policies)
                processor.replace(layer_name, layer, search_inputs=SearchInputs(layer, cur_args, cur_kwargs))

                logger.info("Catching inputs of all Linear in decoder layer.")
                start_time = time.time()

                transform_network_inplace(layer, QuantCell, lambda _, cell: cell.add_hook(self._config.experimental))
                # FIXME: 'always_use_fp_input_in_processer' is a temporary switch for fixing activation between
                # layers. This branch may introduces error to the next layer, because previous processors in the
                # pipeline changes the layer, and thus, gives a inaccurate output. Set the switch to True to
                # avoid this issue. The switch should be removed after the issue is fixed. -- @tongl2
                catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs, do_update= \
                    len(self.decoder_layers) > 1 and not self._config.always_use_fp_input_in_processer)

                transform_network_inplace(layer, QuantCell, lambda _, c: c.remove_hook(self._config.experimental))
                logger.info(f"{i}th layer output refresh time cost {time.time() - start_time}")

                processor.process(layer_name, layer)
                processor.deploy(layer_name, layer)
                network.update_parameters_name()
                gc.collect()
            if self._config.reflash_inputs_after_each_processor:
                catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs)
            start_time = time.time()
            offload_network(layer)
            gc.collect()
            logger.info(f"{i}th layer offload network time cost {time.time() - start_time}")
        return network

    # pylint: disable=unused-argument
    def _apply_mindone(self, model: BaseQuantForCausalLM,
              datasets=None, **kwargs) -> Cell:
        """
        Define how to add fake quantizer to `model` for mindone framework.
        """
        def catch_layer_output(layer, input_args, input_kwargs, output_args, output_kwargs, do_update=True):
            for index, (args, kwargs) in enumerate(zip(input_args, input_kwargs)):
                output = layer(*args, **kwargs)
                if do_update:
                    if "hidden_states" in all_kwargs[index]:
                        output_kwargs[index]["hidden_states"] = output[0] if isinstance(output, tuple) else output
                    else:
                        output_args[index][0] = output[0] if isinstance(output, tuple) else output
        self._config.update_comm_info()
        # convert datasets to Dataset
        datasets = convert_to_dataset(datasets)
        self._check_apply_inputs(datasets)
        catchers = {}
        network = None

        # catch input for different layer, eg: Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock
        # pylint: disable=protected-access
        for layer in model._transformer_layers():
            self.decoder_layer_types.clear()
            self.decoder_layer_types.append(layer)
            self._get_decoder_layers(model.network)
            start_time = time.time()
            logger.info("Catching inputs for first decoder layer with "
                        f"{len(datasets)} datasets samples.")
            catcher, network = self._get_mo_first_layer_input(model, datasets)
            catchers[layer] = catcher

        # get transformer layers from model
        # pylint: disable=protected-access
        for layer in model._transformer_layers():
            self.decoder_layer_types.clear()
            self.decoder_layer_types.append(layer)
            self._get_decoder_layers(model.network)

            start_time = time.time()
            logger.info("Catching inputs for first decoder layer with "
                        f"{len(datasets)} datasets samples.")
            catcher = catchers[layer]
            all_args = catcher.args
            all_kwargs = catcher.kwargs
            logger.info(f"_get_first_layer_input time cost {time.time() - start_time}")
            start_time = time.time()
            logger.info(f"get_decoder_layers time cost {time.time() - start_time}")
            for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running PTQ..."):
                logger.info(f"Quantize {i}th decoder layer.")
                layer_name, layer = self.decoder_layers[i]
                cur_args, cur_kwargs = copy.deepcopy(all_args), copy.deepcopy(all_kwargs)
                catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs,
                                    do_update=len(self.decoder_layers) > 1)
                for processor in PTQ.pipeline:
                    processor = processor(self._config, self.layer_policies)
                    processor.replace(layer_name, layer)

                    logger.info("Catching inputs of all Linear in decoder layer.")
                    start_time = time.time()

                    transform_network_inplace(layer, QuantCell, lambda _, cell: cell.add_hook())
                    catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs, do_update=False)
                    transform_network_inplace(layer, QuantCell, lambda _, c: c.remove_hook())
                    logger.info(f"{i}th layer output refresh time cost {time.time() - start_time}")

                    processor.process(layer_name, layer, quant_model=model,
                                    search_inputs=SearchInputs(layer, cur_args, cur_kwargs))
                    network.update_parameters_name()
                    gc.collect()
                if self._config.reflash_inputs_after_each_processor:
                    catch_layer_output(layer, cur_args, cur_kwargs, all_args, all_kwargs)
                start_time = time.time()
                offload_network(layer)
                gc.collect()
                logger.info(f"{i}th layer offload network time cost {time.time() - start_time}")
        return network

    def fake_quant(self, network):
        """Apply fake quantization to the model.

        This method applies fake quantization to the model, which is useful
        for validating quantization effects without actually converting to
        integer operations.

        Args:
            network (Cell): Network to be fake quantized.

        Returns:
            fake quantized network.

        Raises:
            TypeError: If `network` type is not Cell.
        """
        if not isinstance(network, Cell):
            raise TypeError(f"Input network should be a Cell, but got: {type(Cell)}.")
        self._config.update_comm_info()
        self._get_decoder_layers(network)
        for i in tqdm.tqdm(range(len(self.decoder_layers)), desc="Running PTQ FakeQuant..."):
            layer_name, layer = self.decoder_layers[i]
            for processor in PTQ.pipeline:
                processor = processor(self._config, self.layer_policies)
                processor.set_fake_quant()
                with no_init_parameters():
                    processor.replace(layer_name, layer)
                    processor.deploy(layer_name, layer)
                network.update_parameters_name()
        return network

    def _get_mf_first_layer_input(self, network: Cell, ds=None, helper=None):
        """get mindformers first layer input"""
        catcher = InputCatcher()
        catcher.patch(self.decoder_layers[0][1])
        if not ds:
            raise ValueError("PTQ need dataset to calibrate, please provide dataset.")
        total_count = len(ds)
        data_count = 1
        for _, ds_item in enumerate(ds):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            try:
                if isinstance(helper, NetworkHelper):
                    return helper.generate(network, ds_item['input_ids'], do_sample=False, max_new_tokens=1)
                if self._generate_func is None:
                    return network.generate(**ds_item, do_sample=False, max_new_tokens=1)
                # pylint: disable=not-callable
                return self._generate_func(**ds_item)
            except GeneratorExit:
                if hasattr(network, "block_mgr") and network.block_mgr:
                    network.block_mgr.clear_cache()
            data_count += 1
        catcher.recover()
        offload_network(network)
        return catcher, network

    def _get_mo_first_layer_input(self, model: BaseQuantForCausalLM, ds=None):
        """get mindone first layer input"""
        catcher = InputCatcher()
        catcher.patch(self.decoder_layers[0][1])
        if not ds:
            raise ValueError("PTQ need dataset to calibrate, please provide dataset.")
        total_count = len(ds)
        data_count = 1
        for _, ds_item in enumerate(ds):
            logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
            try:
                # pylint: disable=not-callable
                return model.forward(ds_item)
            except GeneratorExit:
                if hasattr(model.network, "block_mgr") and model.network.block_mgr:
                    model.network.block_mgr.clear_cache()
            data_count += 1
        catcher.recover()
        offload_network(model.network)
        return catcher, model.network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `RoundToNearest.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is ``""``, which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents quantized network.
        """
        logger.info("PTQ.convert take no effect now, no need to invoke.")
        return net_opt

    def _summary_target_layer_type(self) -> tuple:
        return PTQ.target_layer_type

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
