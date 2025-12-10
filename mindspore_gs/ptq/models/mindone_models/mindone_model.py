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
Mindone Quantization Model Base Class
"""
import os
import time
import json
from typing import List
from dataclasses import dataclass, field
from tqdm import tqdm

from mindspore.nn.cell import Cell
from mindspore import load_checkpoint, load_param_into_net

from mindspore_gs.common import logger
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from mindspore_gs.ptq.models.base_model_impl import BaseQuantForCausalLMImpl
from mindspore_gs.ptq.basic_functions.safetensors_mgr import SafeTensorsMgr
from mindspore_gs.ptq.ptq.quant import PTQ
from mindspore_gs.ptq.basic_functions.distributed_parameter import DistributedParameter
from mindspore_gs.ptq.utils import QuantType
from mindspore_gs.common import BackendTarget
from mindspore_gs.common.utils import offload_network, value_check, list_value_check
from .param_processor import ParamProcessor


@dataclass
class SmoothLayerInfo:
    """Data class for storing layer information used in smooth quantization.
    
    This class represents a pair of layers (previous layer and current layer(s))
    that need to be processed together during smooth quantization algorithms
    such as SmoothQuant, OSL (Outlier Suppression Lite), etc.
    
    Attributes:
        prev_layer (Cell): The layer that comes before the current layer(s).
        
        curr_layer (List[Cell]): A list of current layer(s) that will be scaled up.
    
    Note:
        During the quantization process, a 'smooth_scale' field may be added
        to instances of this class (or its dictionary representation) to store
        the computed scaling factors.
    
    Example:
        >>> from mindspore_gs.ptq.models.mindone_models.mindone_model import SmoothLayerInfo
        >>> 
        >>> # Example: QKV projection with input layernorm
        >>> layer_info = SmoothLayerInfo(
        ...     prev_layer=input_layernorm,
        ...     curr_layer=[q_proj, k_proj, v_proj]
        ... )
    
    Raises:
        TypeError: If `prev_layer` is not a Cell instance.
        TypeError: If `curr_layer` is not a list of Cell instances.
    """
    prev_layer: Cell = None
    curr_layer: List[Cell] = field(default_factory=list)

    def __post_init__(self):
        value_check('prev_layer', self.prev_layer, Cell)
        list_value_check('curr_layer', self.curr_layer, Cell)


class MindOneModel(BaseQuantForCausalLMImpl):
    """MindOneModel base class for Quantization"""
    _model_registry: dict[str, type] = {}

    @staticmethod
    def _reg_model(name, model_clazz: type[BaseQuantForCausalLM]):
        cur = MindOneModel._model_registry.get(name)
        if cur:
            raise RuntimeError(f"Duplicated model reg, name: {name}, already reg class: {cur}, "
                               f"current reg class:{model_clazz}")
        logger.info(f"Register mindone model: name {name} to {model_clazz}")
        MindOneModel._model_registry[name] = model_clazz

    @staticmethod
    def reg_model(alias=None):
        def decorator(cls):
            """decorator"""
            register_key = alias if alias is not None else cls.__name__
            MindOneModel._reg_model(register_key, cls)
            return cls

        return decorator

    # pylint: disable=arguments-differ
    @classmethod
    def from_pretrained(cls, model_path):
        """Create a model instance from a pretrained configuration.
        """
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_name = config.get("model_type", "")
        if not model_name:
            raise ValueError(f"No model_type in {config_path}, please check the config file.")
        model_cls = MindOneModel._model_registry.get(model_name, None)
        if model_cls is None:
            raise ValueError(f"Not supported model_name: {model_name} from {model_path}")
        logger.info(f"Create mindone model: {model_name} from pretrained {model_path} with {model_cls}")
        return model_cls(model_path)

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
        raise NotImplementedError

    def forward(self, input_ids, max_new_tokens=1):
        """Perform forward pass through the model.

        This is an abstract method that must be implemented by derived classes.
        It should handle the forward pass logic for model inference.

        Args:
            input_ids (Tensor): Input token IDs for the model.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to ``1``.

        Returns:
            Forward pass results.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def _process_params_dict_before_save(self, quantization_desc, param_dict,
                                         backend=BackendTarget.ASCEND):
        """Process parameter dictionary before saving.
        """
        param_processor = ParamProcessor(backend)
        param_dict = param_processor.process_param_dict(param_dict, quantization_desc)
        return param_dict

    def parameters_dict(self, scope="", backend=BackendTarget.ASCEND):
        param_dict = self.network.parameters_dict()
        quantization_desc = self._get_description_file()
        param_dict = self._process_params_dict_before_save(quantization_desc,
                                                           param_dict,
                                                           backend)
        dis_param_dict = {}
        for name, param in tqdm(param_dict.items(), desc="creating DistributedParameters"):
            dis_param_dict[name] = DistributedParameter(param)
        return dis_param_dict

    def _original_safetensors_path(self):
        """Get the original SafeTensors file path.

        Returns:
            str. Path to the original SafeTensors file.
        """
        return self._original_sf_path

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
        ptq = self._set_ptq_config(ptq, **kwargs)
        quant_start = time.time()
        logger.info('Quantize-ing network...')
        start_time = time.time()
        ptq.apply(self, datasets=datasets, framework="mindone")
        ptq.summary(net)
        offload_network(net)
        logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
        start_time = time.time()
        logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
        logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')

    def _set_ptq_config(self, ptq: PTQ, **kwargs):
        """set ptq config"""
        ptq.set_ptq_config(**kwargs)
        return ptq

    def save_quantized(self, save_path, backend=BackendTarget.ASCEND):
        """Save the quantized model to checkpoint files.

        Args:
            save_path (str): Path where the quantized model should be saved.
            backend (BackendTarget, optional): Target backend for the saved model.
                Defaults to ``BackendTarget.ASCEND``.
        """
        super().save_quantized(save_path, backend)
        sf_mgr = SafeTensorsMgr()
        sf_mgr.save(self._original_sf_path,
                    save_path,
                    self.parameters_dict(backend=backend),
                    self._process_param_desc(backend=backend))

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
        transformer_layers = self._transformer_layers()
        _ = [ptq.decoder_layer_types.append(layer) for layer in transformer_layers]
        ptq.fake_quant(self.network)
        self._load_weights_to_fake_quant(quant_safetensors_path)

    @staticmethod
    def _find_safetensors_file(directory, suffix):
        """Find a safetensors file with the specified suffix in a directory.
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
        return matching_files

    def _load_weights_to_fake_quant(self, quant_safetensors_path):
        """Load weights for fake quantization from checkpoint files.

        Args:
            quant_safetensors_path (str): Path to quantized checkpoint files.
        """
        if not quant_safetensors_path:
            return

        param_dict_paths = self._find_safetensors_file(quant_safetensors_path, ".safetensors")
        param_dict = {}
        for param_dict_path in param_dict_paths:
            param_dict.update(load_checkpoint(param_dict_path, format="safetensors"))
        param_not_load, ckpt_not_load = load_param_into_net(self.network, param_dict)
        logger.info(f"Network has but not in ckpt: {param_not_load}", flush=True)
        logger.info(f"CKPT has but not in network: {ckpt_not_load}", flush=True)

    # pylint: disable=W0221
    def _get_quant_type(self):
        """Get quantization type information for network parameters.

        This method analyzes the network to determine the quantization
        type for each parameter, such as W8A8 or W4A8_DYNAMIC.

        Args:
            network (Cell): The network to analyze for quantization types.

        Returns:
            dict. Dictionary mapping parameter names to their quantization types.

        Raises:
            TypeError: If the input network is not a Cell instance.
        """
        if not isinstance(self.network, Cell):
            raise TypeError(f"Input network should be a Cell, but got: {type(Cell)}.")
        results = {}
        def process(root: Cell, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if not hasattr(cell, "quant_type_dict"):
                    process(cell, full_cell_name)
                    continue
                info = cell.quant_type_dict()
                results.update(info)
        process(self.network, 'network')
        return results

    # pylint: disable=W0221
    def _get_description_file(self):
        """Obtain the description of quantization type for network parameters.

        This method generates a comprehensive description of the
        quantization type for each parameter in each layer of the network.
        The description includes information such as W8A8 or W4A8_DYNAMIC
        for each parameter.

        Args:
            network (Cell): The network to analyze for quantization descriptions.

        Returns:
            dict. Dictionary mapping parameter names to their quantization
                type descriptions.
        """
        results = self._get_quant_type()
        param_dict = self.network.parameters_dict()

        desc_info = {}
        for key in param_dict:
            if key in results:
                desc_info[key] = results[key]
            else:
                desc_info[key] = QuantType.FLOAT.value
        return desc_info

    def _process_param_desc(self, backend=BackendTarget.ASCEND):
        """process param description."""
        param_processor = ParamProcessor(backend)
        param_desc = param_processor.process_param_desc(self._get_description_file())
        return param_desc

    def get_layers_for_smooth(self, decoder_layer: Cell) -> List[SmoothLayerInfo]:
        """Get layer pairs for smooth quantization algorithms.
        
        This method identifies and returns a list of layer pairs within a decoder layer
        that need to be processed together during smooth quantization algorithms such as
        SmoothQuant, OSL (Outlier Suppression Lite), etc.
        
        Each returned `SmoothLayerInfo` represents a pair of layers where:
        - The `prev_layer` (previous layer) will have its output scaled down
        - The `curr_layer` (current layer(s)) will have their weights scaled up
        
        This scaling operation helps to reduce quantization errors by balancing the
        dynamic ranges between layers.
        
        Args:
            decoder_layer (Cell): A single transformer decoder layer instance from the model.
                This should be an instance of the model's decoder layer class (e.g.,
                `Glm4vTextDecoderLayer`, `Qwen3DecoderLayer`). The method will extract
                layer pairs from this decoder layer for smooth quantization processing.
        
        Returns:
            List[SmoothLayerInfo]: A list of `SmoothLayerInfo` objects, each representing
                a layer pair to be processed for smooth quantization. The list typically
                includes pairs for:
                
                - **Attention layers**: 
                  - QKV projection: `prev_layer=input_layernorm`, `curr_layer=[q_proj, k_proj, v_proj]`
                  - Output projection: `prev_layer=v_proj`, `curr_layer=[o_proj]`
                
                - **MLP layers**:
                  - Gate/Up projection: `prev_layer=post_attention_layernorm`, `curr_layer=[gate_proj, up_proj]`
                  - Down projection: `prev_layer=up_proj` (or `gate_up_proj`), `curr_layer=[down_proj]`
                
                Each `SmoothLayerInfo` contains:
                    - `prev_layer` (Cell): The previous layer whose output will be scaled down.
                        Typically a normalization layer (e.g., `input_layernorm`, 
                        `post_attention_layernorm`) or a projection layer.
                    - `curr_layer` (List[Cell]): A list of one or more current layers whose
                        weights will be scaled up.
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses to provide
                model-specific layer pair definitions.
            TypeError: If `decoder_layer` is not a `Cell` instance.
            ValueError: If the layer structure in `decoder_layer` does not match the expected
                model architecture.
        
        Example:
            >>> # In a subclass implementation (e.g., GLM4v)
            >>> def get_layers_for_smooth(self, decoder_layer):
            ...     layers_info = []
            ...     # Attention QKV projection
            ...     layers_info.append(
            ...         SmoothLayerInfo(
            ...             prev_layer=decoder_layer.input_layernorm,
            ...             curr_layer=[
            ...                 decoder_layer.self_attn.q_proj,
            ...                 decoder_layer.self_attn.k_proj,
            ...                 decoder_layer.self_attn.v_proj
            ...             ]
            ...         )
            ...     )
            ...     # Attention output projection
            ...     layers_info.append(
            ...         SmoothLayerInfo(
            ...             prev_layer=decoder_layer.self_attn.v_proj,
            ...             curr_layer=[decoder_layer.self_attn.o_proj]
            ...         )
            ...     )
            ...     return layers_info
        
        Note:
            - The order of `SmoothLayerInfo` objects in the returned list matters, as
              smooth quantization algorithms typically process them sequentially.
            - The `prev_layer` and `curr_layer` must be actual layer instances from
              the `decoder_layer`, not copies or references to other layers.
            - During the quantization process, a `smooth_scale` field may be dynamically
              added to each `SmoothLayerInfo` to store the computed scaling factors.
        """
        raise NotImplementedError

    def _get_gqa_info(self, model_path):
        """Get GQA information from config file."""
        config_path = os.path.join(model_path, 'config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            text_config = config.get('text_config', None)
            if text_config is None:
                text_config = config
            num_attention_heads = text_config.get('num_attention_heads', None)
            num_key_value_heads = text_config.get('num_key_value_heads', None)
            if num_attention_heads is None or num_key_value_heads is None:
                raise ValueError(f"num_attention_heads or num_key_value_heads is not found in {config_path}.")
            return num_attention_heads, num_key_value_heads
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config file not found at {config_path}.") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {config_path}.") from e
