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
from tqdm import tqdm


from mindspore import load_checkpoint, load_param_into_net

from mindspore_gs.common import logger
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM
from mindspore_gs.ptq.models.base_model_impl import BaseQuantForCausalLMImpl
from mindspore_gs.ptq.basic_functions.safetensors_mgr import SafeTensorsMgr
from mindspore_gs.ptq.ptq.quant import PTQ
from mindspore_gs.ptq.basic_functions.distributed_parameter import DistributedParameter
from mindspore_gs.common import BackendTarget
from mindspore_gs.common.utils import offload_network
from .param_processor import ParamProcessor


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
        raise NotImplementedError

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
        param_processor = ParamProcessor(backend, quantization_desc)
        param_dict = param_processor.deploy(param_dict)
        return param_dict

    def parameters_dict(self, scope="", backend=BackendTarget.ASCEND):
        param_dict = self.network.parameters_dict()
        quantization_desc = self.get_description_file()
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

    def save_quantized(self, save_path):
        """Save the quantized model to checkpoint files.

        Args:
            save_path (str): Path where the quantized model should be saved.
        """
        super().save_quantized(save_path)
        sf_mgr = SafeTensorsMgr()
        sf_mgr.save(self._original_sf_path,
                    save_path,
                    self.parameters_dict(backend=BackendTarget.ASCEND),
                    self.get_description_file())

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
    def get_description_file(self):
        """Obtain the description of quantization type for each parameter.

        This method generates a description file that maps each network
        parameter to its quantization type (e.g., W8A8, W4A8_DYNAMIC).
        This information is useful for understanding the quantization
        characteristics of different parts of the model.

        Args:
            network (Cell): The network to analyze for quantization descriptions.

        Returns:
            Description of quantization types for network parameters.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
