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

"""
MindFormers Quantization Model Base Class

This module provides the base implementation for quantizing models
using the MindFormers framework. It extends the generic quantization
base classes to provide specific functionality for MindFormers models.

The MFModel class serves as the foundation for all MindFormers-specific
quantized model implementations. It handles:
- Model loading and initialization from MindFormers configurations
- Integration with MindFormers' distributed computing capabilities
- Parameter management compatible with MindFormers' tensor parallelism
- SafeTensors format support for efficient model saving and loading

This implementation is designed to work seamlessly with MindFormers'
model zoo and supports various large language models including Qwen3,
DeepSeekV3, and other transformer-based architectures.

Examples:
    >>> from mindspore_gs.ptq.models.mindformers_models import MFModel
    >>>
    >>> # The class is typically used through specific implementations
    >>> # like QWen3, DeepSeekV3, etc.
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/model.yaml")
"""

import os
import time
import json
from tqdm import tqdm
import mindspore as ms
from mindspore import Parameter, ops as msops
from mindspore.communication import get_rank
from mindspore.nn.utils import no_init_parameters
from mindspore import load_param_into_net, load_checkpoint
from mindformers import AutoModel, MindFormerConfig, build_context, build_parallel_config
from mindformers.parallel_core.inference.tensor_parallel.layers import (RowParallelLinear,
                                                                        ColumnParallelLinear,
                                                                        MergedColumnParallelLinear,
                                                                        QKVParallelLinear)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (ColumnParallelGroupedLinear,
                                                                                RowParallelGroupedLinear)
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from mindspore_gs.ptq.models.base_model_impl import BaseQuantForCausalLMImpl
from mindspore_gs.common import logger
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQ
from mindspore_gs.ptq.basic_functions.distributed_parameter import DistributedParameter
from mindspore_gs.ptq.basic_functions.processor import Processor
from mindspore_gs.ptq.quant_cells.mindformers.mcore_linear_wrapper import McoreLinearInferCell
from mindspore_gs.ptq.basic_functions.safetensors_mgr import SafeTensorsMgr
from mindspore_gs.common.utils import offload_network


class MFModel(BaseQuantForCausalLMImpl):
    """MindFormers Model Base Class for Quantization

    This class provides the base implementation for quantizing models
    using the MindFormers framework. It extends BaseQuantForCausalLMImpl
    to provide specific functionality for MindFormers models.

    Key features of this implementation include:
    - Seamless integration with MindFormers model configurations
    - Support for distributed computing and tensor parallelism
    - Efficient parameter management for large-scale models
    - SafeTensors format support for model persistence
    - Compatibility with MindFormers' model zoo

    The class uses a registry pattern to allow specific model implementations
    (like Qwen3, DeepSeekV3) to register themselves and be automatically
    discovered by the AutoQuantForCausalLM interface.

    Examples:
        >>> # Typically used through specific implementations like QWen3
        >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
        >>>
        >>> # Automatically selects the appropriate MindFormers implementation
        >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/qwen3_config.yaml")
    """
    _model_registry: dict[str, type] = {}

    @staticmethod
    def _reg_model(name, model_clazz: type[BaseQuantForCausalLM]):
        cur = MFModel._model_registry.get(name)
        if cur:
            raise RuntimeError(f"Duplicated model reg, name: {name}, already reg class: {cur}, "
                               f"current reg class:{model_clazz}")
        logger.info(f"Register mindformers model: name {name} to {model_clazz}")
        MFModel._model_registry[name] = model_clazz

    @staticmethod
    def reg_model(alias=None):
        def decorator(cls):
            """decorator"""
            register_key = alias if alias is not None else cls.__name__
            MFModel._reg_model(register_key, cls)
            return cls

        return decorator

    def __init__(self, yaml_path):
        """
        Initialize the MindFormers quantized model.

        This method initializes the model by loading the MindFormers
        configuration, building the execution context, and creating
        the underlying network instance.

        Args:
            yaml_path (str): Path to the MindFormers model configuration YAML file.
        """
        config = MindFormerConfig(yaml_path)
        build_context(config)
        build_parallel_config(config)
        with no_init_parameters():
            self.network = AutoModel.from_config(yaml_path)

        self._original_sf_path = config.pretrained_model_dir
        if not self._original_sf_path:
            raise ValueError(f"Make sure pretrained_model_dir in yaml-file is not empty: {yaml_path}")
        self.network.load_weights(self._original_sf_path)
        self._after_network_load_weights()

    # pylint: disable=arguments-differ
    @classmethod
    def from_pretrained(cls, yaml_path):
        """
        Create a model instance from a pretrained configuration.

        This method creates a model instance by loading the MindFormers
        configuration and selecting the appropriate specific model
        implementation based on the configuration.

        Args:
            yaml_path (str): Path to the MindFormers model configuration YAML file.

        Returns:
            MFModel. An instance of the appropriate model implementation.

        Raises:
            ValueError: If the model name in the configuration is not supported.
        """
        if not os.path.isfile(yaml_path):
            raise ValueError(f"The {yaml_path} is not exists, "
                             "please check the yaml path.")
        if not yaml_path.endswith('.yaml'):
            raise ValueError(f"The {yaml_path} is not a yaml file, "
                             "please check the yaml path.")
        config = MindFormerConfig(yaml_path)
        if not hasattr(config, 'trainer') or not hasattr(config.trainer, 'model_name'):
            raise ValueError(f"Not contain trainer.model_name in yaml-file: {yaml_path}")
        model_name = config.trainer.model_name
        model_cls = MFModel._model_registry.get(model_name, None)
        if model_cls is None:
            raise ValueError(f"Not supported model_name: {model_name} from yaml: {yaml_path}")
        logger.info(f"Create mindformers model: {model_name} from yaml: {yaml_path} with {model_cls}")
        return model_cls(yaml_path)

    def _original_safetensors_path(self):
        """Get the original SafeTensors file path.

        Returns:
            str. Path to the original SafeTensors file.
        """
        return self._original_sf_path

    def forward(self, input_ids, max_new_tokens=1):
        """Perform forward pass through the model.

        This method delegates to the underlying MindFormers network's
        generate method for inference.

        Args:
            input_ids (Tensor): Input token IDs for the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to ``1``.

        Returns:
            Generated output from the model.
        """
        return self.network.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)

    def calibrate(self, ptq_config, layers_policy, datasets, **kwargs):
        """Calibrate and quantize the model.

        This method implements the core quantization workflow including:
        1. Setting up the PTQ algorithm with the provided configuration
        2. Applying the quantization to the network
        3. Performing calibration using the provided datasets
        4. Managing timing and performance monitoring

        Args:
            ptq_config (PTQConfig): Configuration for post-training quantization.
            layers_policy (dict): Policy for different layer quantization strategies.
            datasets (Dataset): Calibration dataset for quantization.
            **kwargs: Additional keyword arguments.
                fake_quant (bool, optional): Whether to use fake quantization.
                    Defaults to ``False``.

        Example:
            >>> # Typical usage pattern
            >>> model.calibrate(
            ...     ptq_config=ptq_config,
            ...     layers_policy=layers_policy,
            ...     datasets=calibration_dataset,
            ...     fake_quant=False
            ... )
        """
        logger.info("Use ptq algo to quant network and weight.")
        net = self._network()
        ptq = PTQ(config=ptq_config, layer_policies=layers_policy)
        # pylint: disable=protected-access
        ptq = self._set_ptq_config(ptq, **kwargs)
        ptq = self._load_mindformers_plugin(ptq)
        quant_start = time.time()
        logger.info('Quantize-ing network...')
        start_time = time.time()
        ptq.apply(net, datasets=datasets)
        ptq.summary(net)
        offload_network(net)
        logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
        start_time = time.time()
        logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
        logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')

    def _set_ptq_config(self, ptq: PTQ, **kwargs):
        """set ptq config"""
        ptq.set_ptq_config(experimental=True)
        ptq.set_ptq_config(**kwargs)
        return ptq

    def _load_mindformers_plugin(self, ptq: PTQ):
        """load mindformers plugin"""
        # set decoder layer types
        transformer_layers = self._transformer_layers()
        _ = [ptq.decoder_layer_types.append(layer) for layer in transformer_layers]
        # set generate function for getting first layer input
        ptq.set_generate_func(self.forward)
        return ptq

    def _network(self):
        """Get the underlying network instance.

        Returns:
            The underlying MindFormers network instance.
        """
        return self.network

    def _transformer_layers(self) -> tuple[type]:
        """Get the transformer layer types for quantization.

        This method returns the transformer layer types that should
        be targeted for quantization in MindFormers models.

        Returns:
            tuple[type]. Tuple containing TransformerLayer type.
        """
        from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
        return [TransformerLayer]

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        """Process parameter dictionary before saving.

        This method filters out certain parameters that should not be
        saved, such as cache and float weight parameters.

        Args:
            param_dict (dict): Dictionary of model parameters.

        Returns:
            tuple[dict, dict]. Tuple containing the filtered parameter
                dictionary and parameter name trace.
        """
        new_param_dict = {}
        for key, param in param_dict.items():
            if "key_cache" in key or "value_cache" in key or "float_weight" in key:
                continue
            new_param_dict[key] = param
        return new_param_dict, {}

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        """Load weights for fake quantization.

        This is an abstract method that must be implemented by derived classes.

        Args:
            quant_safetensors_path (str): Path to quantized SafeTensors file.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _after_network_load_weights(self):
        return

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        """Apply fake quantization to the model.

        This method applies fake quantization to the model, inserting
        quantization and dequantization operations in the computation
        graph while keeping the underlying operations in floating point.

        Args:
            ptq_config (PTQConfig): Configuration for post-training quantization.
            layers_policy (dict): Policy for different layer quantization strategies.
            quant_safetensors_path (str, optional): Path to quantized SafeTensors file.
                Defaults to ``""``.
        """
        logger.info("Use ptq algo to fake-quant network and weight")
        ptq = PTQ(config=ptq_config, layer_policies=layers_policy)
        # pylint: disable=protected-access
        ptq._config.experimental = True
        ptq._config.fake_quant = True
        transformer_layers = self._transformer_layers()
        _ = [ptq.decoder_layer_types.append(layer) for layer in transformer_layers]
        ptq.fake_quant(self.network)
        self._load_weights_to_fake_quant(quant_safetensors_path)


class MFModelEnableSafeTensors(MFModel):
    """MindFormers Model with SafeTensors Support

    This class extends MFModel to provide support for SafeTensors
    format for efficient model saving and loading. SafeTensors is
    a format that provides faster loading times and better security
    compared to traditional checkpoint formats.
    """

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        """Load weights for fake quantization from SafeTensors.

        This method loads quantized weights from SafeTensors files
        for fake quantization.

        Args:
            quant_safetensors_path (str): Path to quantized SafeTensors file.
        """
        from .weight_loader import WeightProcessor
        processor = WeightProcessor()
        processor.load_safetensors_shard(quant_safetensors_path, self.network, self._convert_param_names_to_hf)

    def _process_params_dict_before_save(self, param_dict) -> tuple[dict, dict]:
        """Process parameter dictionary before saving to SafeTensors.

        This method filters parameters and handles expert weights
        specifically for SafeTensors format.

        Args:
            param_dict (dict): Dictionary of model parameters.

        Returns:
            tuple[dict, dict]. Tuple containing the processed parameter
                dictionary and parameter name trace.
        """
        param_dict, param_name_trace = super()._process_params_dict_before_save(param_dict)
        # _del_experts_weight
        experts_dict = {k: v for k, v in param_dict.items()
                        if ".mlp.experts." in k}
        is_fc1_quant = any(".linear_fc1.weight_scale" in k for k in experts_dict.keys())
        is_fc2_quant = any(".linear_fc2.weight_scale" in k for k in experts_dict.keys())
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if is_fc1_quant and hasattr(cell, "weight1"):
                    del cell.weight1
                    cell.weight1 = None
                if is_fc2_quant and hasattr(cell, "weight2"):
                    del cell.weight2
                    cell.weight2 = None
                process(cell, full_cell_name)
        process(self.network, 'network')

        new_param_dict = {}
        for key, value in param_dict.items():
            if (is_fc1_quant and "weight1" in key) or \
                (is_fc2_quant and "weight2" in key):
                continue
            new_param_dict[key] = value
        return new_param_dict, param_name_trace

    def _shard_dict(self):
        """Generate sharding dictionary for distributed parameters.

        This method creates a dictionary that maps parameter names
        to their sharding axes, which is used for distributed computing.

        Returns:
            dict. Dictionary mapping parameter names to sharding axes.
        """
        class Collector(Processor):
            """Collector for parameter sharding information."""
            def __init__(self):
                self.shard_axis = {}
                self.row_linears = ('linear_proj', 'linear_fc2')
                self.col_linears = {'linear_qkv', 'linear_fc1',
                                    'linear_q', 'linear_k', 'linear_v'}

            @staticmethod
            def _transpose_b(linear):
                """Check if linear layer transposes the weight matrix."""
                if isinstance(linear, (RowParallelLinear, ColumnParallelLinear,
                                       QKVParallelLinear, MergedColumnParallelLinear)):
                    return linear.transpose_b
                if isinstance(linear, (ColumnParallelGroupedLinear, RowParallelGroupedLinear)):
                    return False
                raise ValueError(f"Not supported linear: {type(linear)}")

            def _try_append_shard_axis(self, linear, param_name, axis):
                """Append sharding axis information for a parameter."""
                if not hasattr(linear, param_name) or getattr(linear, param_name) is None:
                    return
                self.shard_axis[getattr(linear, param_name).name] = axis

            def process_cell(self, cell_name, cell):
                """Process a network cell to collect sharding information."""
                if 'linear_proj' in cell_name:
                    # pylint: disable=protected-access
                    transpose_b = cell._transpose_b()
                    self._try_append_shard_axis(cell, 'weight', 1 if transpose_b else 0)
                    self._try_append_shard_axis(cell, 'weight_scale', None)
                    self._try_append_shard_axis(cell, 'weight_offset', None)
                    self._try_append_shard_axis(cell, 'input_scale', 0)
                    self._try_append_shard_axis(cell, 'input_offset', 0)
                    self._try_append_shard_axis(cell, 'smooth_scale', 0)
                    self._try_append_shard_axis(cell, 'deq_scale', None)
                    self._try_append_shard_axis(cell, 'quant_bias', None)
                    self._try_append_shard_axis(cell, 'bias', None)
                elif 'linear_fc2' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 1)
                    self._try_append_shard_axis(cell, 'weight_scale', None)
                    self._try_append_shard_axis(cell, 'weight_offset', None)
                    self._try_append_shard_axis(cell, 'input_scale', 0)
                    self._try_append_shard_axis(cell, 'input_offset', 0)
                    self._try_append_shard_axis(cell, 'smooth_scale', 0)
                    self._try_append_shard_axis(cell, 'deq_scale', None)
                    self._try_append_shard_axis(cell, 'quant_bias', None)
                    self._try_append_shard_axis(cell, 'bias', None)
                elif any(seg in cell_name for seg in ('linear_q', 'linear_k',
                                                      'linear_v', 'linear_qkv')):
                    # pylint: disable=protected-access
                    transpose_b = cell._transpose_b()
                    self._try_append_shard_axis(cell, 'weight', 0 if transpose_b else 1)
                    self._try_append_shard_axis(cell, 'weight_scale', 0)
                    self._try_append_shard_axis(cell, 'weight_offset', 0)
                    self._try_append_shard_axis(cell, 'input_scale', None)
                    self._try_append_shard_axis(cell, 'input_offset', None)
                    self._try_append_shard_axis(cell, 'smooth_scale', None)
                    self._try_append_shard_axis(cell, 'deq_scale', 0)
                    self._try_append_shard_axis(cell, 'quant_bias', 0)
                    self._try_append_shard_axis(cell, 'bias', 0)
                elif any(seg in cell_name for seg in ('hidden', 'gating',
                                                      'linear_fc1')):
                    self._try_append_shard_axis(cell, 'weight', 0)
                    self._try_append_shard_axis(cell, 'weight_scale', 0)
                    self._try_append_shard_axis(cell, 'weight_offset', 0)
                    self._try_append_shard_axis(cell, 'input_scale', None)
                    self._try_append_shard_axis(cell, 'input_offset', None)
                    self._try_append_shard_axis(cell, 'smooth_scale', None)
                    self._try_append_shard_axis(cell, 'deq_scale', 0)
                    self._try_append_shard_axis(cell, 'quant_bias', 0)
                    self._try_append_shard_axis(cell, 'bias', 0)
                elif 'output_layer' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 0)
                elif 'embedding.word_embeddings' in cell_name:
                    self._try_append_shard_axis(cell, 'weight', 0)
                else:
                    pass
                if isinstance(cell, McoreLinearInferCell):
                    return cell, True
                return cell, False

        collector = Collector()
        collector.process(self.network)
        return collector.shard_axis

    def parameters_dict(self, scope="") -> dict[str, DistributedParameter]:
        """Get the dictionary of model parameters with distributed information.

        This method returns a dictionary mapping parameter names to
        DistributedParameter objects that include sharding information.

        Args:
            scope (str, optional): Scope for parameter retrieval. Defaults to ``""``.

        Returns:
            dict[str, DistributedParameter]. Dictionary mapping parameter names
            to DistributedParameter objects with sharding information.
        """
        param_dict = self.network.parameters_dict()
        logger.debug(f"network original param_dict: {param_dict}")
        param_dict, param_name_trace = self._process_params_dict_before_save(param_dict)
        logger.debug(f"network param_dict after process: {param_dict}")
        shard_info = self._shard_dict()
        logger.debug(f"network shard info: {shard_info}")
        dis_param_dict = {}
        for name, param in tqdm(param_dict.items(), desc="creating DistributedParameters"):
            shard_axis = shard_info.get(name)
            old_name = name
            while shard_axis is None:
                logger.debug(f"param_name_trace searching key {old_name}")
                old_name = param_name_trace.get(old_name)
                logger.debug(f"param_name_trace searching value {old_name}")
                if old_name is None:
                    break
                shard_axis = shard_info.get(old_name)
            logger.debug("shard axis for ", name, ' is ', shard_axis)
            if shard_axis is None:
                dis_param_dict[name] = DistributedParameter(param)
            else:
                dis_param_dict[name] = DistributedParameter(param, shard_axis)
        hf_param_dict = {}
        for name, param in dis_param_dict.items():
            hf_param_dict[self._convert_param_names_to_hf(name)] = param
        return hf_param_dict

    def save_quantized(self, save_path, backend=BackendTarget.ASCEND):
        """Save the quantized model in SafeTensors format.

        This method saves the quantized model parameters and metadata
        in SafeTensors format for efficient loading and better security.

        Args:
            save_path (str): Path where the quantized model should be saved.
            backend (BackendTarget, optional): Target backend for the saved model.
                Defaults to ``BackendTarget.ASCEND``.
        """
        if backend != BackendTarget.ASCEND:
            raise ValueError("Only support save quantized model for ASCEND backend "
                             "when enable SafeTensors format in mindformers models.")
        super().save_quantized(save_path, backend=backend)
        sf_mgr = SafeTensorsMgr()
        sf_mgr.save(self._original_sf_path,
                    save_path,
                    self.parameters_dict(),
                    self._get_description_file(self._network()))

    def _get_description_file(self, network):
        """Get the description file for quantization information.

        This is an abstract method that must be implemented by derived classes.

        Args:
            network: The network to analyze for quantization descriptions.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @classmethod
    def _convert_param_names_to_hf(cls, param_name):
        """Convert the parameter to huggingface format.

        This is an abstract method that must be implemented by derived classes.

        Args:
            param_name: The parameter name to convert to huggingface format.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class MFModelNotEnableSafeTensors(MFModel):
    """MindFormers Model without SafeTensors Support

    This class provides the same functionality as MFModelEnableSafeTensors
    but without SafeTensors format support, using traditional checkpoint
    formats instead.
    """

    @staticmethod
    def _find_unique_file(directory, suffix):
        """Find a unique file with the specified suffix in a directory.

        Args:
            directory (str): Directory to search in.
            suffix (str): File suffix to look for.

        Returns:
            str. Path to the unique file found.

        Raises:
            FileNotFoundError: If the directory doesn't exist.
            ValueError: If no file or multiple files with the suffix are found.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"directory not exist: {directory}")

        matching_files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith(suffix):
                matching_files.append(file_path)

        if not matching_files:
            raise ValueError(f"not found any 'xxx.{suffix}' file under {directory}")
        if len(matching_files) > 1:
            error_msg = f"found multi 'xxx.{suffix}' file under {directory}: {matching_files}"
            raise ValueError(error_msg)

        return matching_files[0]

    def _concat_route_moe_weight(self, param_dict) -> dict:
        """Concatenate routed MoE weights.

        This method handles the concatenation of weights for Mixture of
        Experts models with routing mechanisms.

        Args:
            param_dict (dict): Dictionary of model parameters.

        Returns:
            dict. Dictionary with concatenated MoE weights.
        """
        new_param_dict = {}
        experts_dict = {k: v for k, v in param_dict.items()
                        if ".mlp.experts." in k}
        other_dict = dict(param_dict.items() - experts_dict.items())
        new_param_dict.update(other_dict)
        is_fc1_quant = any(".linear_fc1.weight_scale" in k for k in experts_dict.keys())
        is_fc2_quant = any(".linear_fc2.weight_scale" in k for k in experts_dict.keys())

        experts_fc1_dict = {k: v for k, v in experts_dict.items()
                            if ".mlp.experts" in k and ".linear_fc1" in k}
        experts_fc1_dict = self._concat_experts(experts_fc1_dict, is_fc1_quant, "weight1")

        experts_fc2_dict = {k: v for k, v in experts_dict.items()
                            if ".mlp.experts" in k and ".linear_fc2" in k}
        experts_fc2_dict = self._concat_experts(experts_fc2_dict, is_fc2_quant, "weight2")

        new_param_dict.update(experts_fc1_dict)
        new_param_dict.update(experts_fc2_dict)
        return new_param_dict, is_fc1_quant, is_fc2_quant

    def _concat_experts(self, param_dict, is_quant, weight_name):
        """Concatenate expert weights for MoE models.

        Args:
            param_dict (dict): Dictionary of expert parameters.
            is_quant (bool): Whether quantization is applied.
            weight_name (str): Name of the weight parameter.

        Returns:
            dict. Dictionary with concatenated expert weights.
        """
        new_param_dict = {}
        for key, _ in param_dict.items():
            key_split = key.split('.')
            prefix_str = '.'.join(key_split[:6])
            suffix_str = '.'.join(key_split[7:])
            if is_quant:
                new_name = f"{prefix_str}.{suffix_str}"
            else:
                new_name = f"{prefix_str}.{weight_name}"
            if new_name in new_param_dict:
                continue
            experts_dict = {k: v for k, v in param_dict.items()
                            if k.startswith(prefix_str) and k.endswith(suffix_str)}
            num_experts = len(experts_dict)
            value_list = []
            for i in range(num_experts):
                key_ = f"{prefix_str}.{i}.{suffix_str}"
                value_ = experts_dict[key_]
                if key_.endswith('.weight'):
                    value_ = msops.transpose(value_, (1, 0))
                value_ = value_.expand_dims(0)
                value_list.append(value_)
            new_value = msops.cat(tuple(value_list), axis=0)
            new_param_dict[new_name] = Parameter(new_value)
        return new_param_dict

    def _del_experts_weight(self, network, is_fc1_quant, is_fc2_quant):
        """Delete expert weights after processing.

        Args:
            network: The network instance.
            is_fc1_quant (bool): Whether FC1 is quantized.
            is_fc2_quant (bool): Whether FC2 is quantized.
        """
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if is_fc1_quant and hasattr(cell, "weight1"):
                    del cell.weight1
                if is_fc2_quant and hasattr(cell, "weight2"):
                    del cell.weight2
                process(cell, full_cell_name)
        process(network, 'network')

    def _process_params_dict_before_load(self, param_dict) -> dict:
        """Process parameter dictionary before loading.

        Args:
            param_dict (dict): Dictionary of model parameters.

        Returns:
            dict. Processed parameter dictionary.
        """
        param_dict, is_fc1_quant, is_fc2_quant = self._concat_route_moe_weight(param_dict)
        self._del_experts_weight(self.network, is_fc1_quant, is_fc2_quant)
        return param_dict

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        """Load weights for fake quantization from checkpoint files.

        Args:
            quant_safetensors_path (str): Path to quantized checkpoint files.
        """
        if not quant_safetensors_path:
            return
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        param_dict_path = os.path.join(quant_safetensors_path, f"rank_{rank_id}")
        param_dict_path = MFModelNotEnableSafeTensors._find_unique_file(param_dict_path, ".safetensors")
        param_dict = load_checkpoint(param_dict_path, format="safetensors")
        new_param_dict = self._process_params_dict_before_load(param_dict)
        param_not_load, ckpt_not_load = load_param_into_net(self.network, new_param_dict)
        logger.info(f"Network has but not in ckpt: {param_not_load}", flush=True)
        logger.info(f"CKPT has but not in network: {ckpt_not_load}", flush=True)

    def parameters_dict(self, scope=""):
        """Get the dictionary of model parameters.

        Args:
            scope (str, optional): Scope for parameter retrieval. Defaults to ``""``.

        Returns:
            dict. Dictionary of model parameters.
        """
        param_dict = self.network.parameters_dict()
        param_dict, _ = self._process_params_dict_before_save(param_dict)
        return param_dict

    def save_quantized(self, save_path, backend=BackendTarget.ASCEND):
        """Save the quantized model to checkpoint files.

        Args:
            save_path (str): Path where the quantized model should be saved.
            backend (BackendTarget, optional): Target backend for the saved model.
                Defaults to ``BackendTarget.ASCEND``.
        """
        if backend != BackendTarget.ASCEND:
            raise ValueError("Only support save quantized model for ASCEND backend "
                             "when not enable SafeTensors format in mindformers models.")
        super().save_quantized(save_path, backend)
        self._save_safetenors(save_path)
        _ = self._save_desc_json(save_path)

    def _save_safetenors(self, save_path) -> str:
        """Save model parameters in SafeTensors format.

        Args:
            save_path (str): Path where parameters should be saved.

        Returns:
            str. Path to the saved SafeTensors file.
        """
        start = time.time()
        logger.info("Saving checkpoint...", flush=True)
        param_dict = self.parameters_dict()
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_path = os.path.join(save_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, 'quant')
        ms.save_checkpoint(param_dict, final_path, format="safetensors")
        logger.info(f'Checkpoint saved to {final_path}', flush=True)
        logger.info(f'Save checkpoint cost time is {time.time() - start} s.')

    def _save_desc_json(self, save_path) -> str:
        """Save quantization description JSON file.

        Args:
            save_path (str): Path where the description file should be saved.

        Returns:
            str. Path to the saved description JSON file.
        """
        start = time.time()
        logger.info("Saving describle json file...", flush=True)
        desc_info = self._get_description_file(self._network())
        save_json_path = os.path.join(save_path, "quantization_description.json")
        os.makedirs(save_path, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(desc_info, f, ensure_ascii=False, indent=4)
        logger.info(f'Describle json file saved to {save_json_path}', flush=True)
        logger.info(f'Save describle json cost time is {time.time() - start} s.')
        return save_json_path

    def _get_description_file(self, network):
        """Get the description file for quantization information.

        This is an abstract method that must be implemented by derived classes.

        Args:
            network: The network to analyze for quantization descriptions.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
