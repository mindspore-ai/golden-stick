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

"""Network helper base class."""
from typing import Union, List

import enum
from dataclasses import dataclass
import numpy as np
from mindspore.nn import Cell

from mindspore_gs.common.utils import value_check


class LayerType(enum.Enum):
    """
    Pre layer type Enum.

    - ``UNKNOWN`` : pre layer type is unknown.
    - ``NORM_LAYER`` : pre layer type is norm layer.
    - ``LINEAR_LAYER`` : pre layer type is linear layer.
    - ``CONCAT_LINEAR_LAYER`` : pre layer type is qkv concat linear layer.
    """
    UNKNOWN = 0
    NORM_LAYER = 1
    LINEAR_LAYER = 2
    CONCAT_LINEAR_LAYER = 3


@dataclass
class LayerInfo:
    """
    Dataclass for recording layer information.

    Args:
        name (str) - name of layer。
        layer (Cell) - layer。
        type_ (LayerType) - type of layer, ``NORM_LAYER``is norm layer,
            ``LINEAR_LAYER`` is linear,``CONCAT_LINEAR_LAYER``is qkv concat linear layer,
            ``UNKNOWN``is unknown type.

    Raises:
        TypeError: `name` is not str.
        TypeError: `layer` type is not Cell.
        TypeError: `type_` not in [LayerType.UNKNOWN, LayerType.NORM_LAYER, LayerType.LINEAR_LAYER,
            LayerType.CONCAT_LINEAR_LAYER].

    Example:
        >>> from mindspore_gs.ptq.network_helpers import LayerInfo, LayerType
        >>> LayerInfo(name='model.layers.0.w_qkv', layer=layer, type=LayerType.CONCAT_LINEAR_LAYER)
    """
    name: str = ""
    layer: Cell = None
    type_: LayerType = LayerType.UNKNOWN

    def __post_init__(self):
        value_check('name', self.name, str)
        value_check('layer', self.layer, Cell)
        value_check('type', self.type_, LayerType)


class DecoderGroupInfo:
    """DecoderGroupInfo."""
    def __init__(self, name, decoder, **kwargs):
        self.decoder: LayerInfo = LayerInfo(name, decoder)
        self.qkv_concat: bool = kwargs.get("qkv_concat", False)
        self.ffn_concat: bool = kwargs.get("ffn_concat", False)
        self.attention_norm: LayerInfo = kwargs.get("attention_norm", None)
        self.attention: LayerInfo = kwargs.get("attention", None)
        self.qkv_mm: LayerInfo = kwargs.get("qkv_mm", None)
        self.q_mm: LayerInfo = kwargs.get("q_mm", None)
        self.k_mm: LayerInfo = kwargs.get("k_mm", None)
        self.v_mm: LayerInfo = kwargs.get("v_mm", None)
        self.o_mm: LayerInfo = kwargs.get("o_mm", None)
        self.ffn_norm: LayerInfo = kwargs.get("ffn_norm", None)
        self.ffn: LayerInfo = kwargs.get("ffn", None)
        self.gate_hidden_mm: LayerInfo = kwargs.get("gate_hidden_mm", None)
        self.gate_mm: LayerInfo = kwargs.get("gate_mm", None)
        self.hidden_mm: LayerInfo = kwargs.get("hidden_mm", None)
        self.w2_mm: LayerInfo = kwargs.get("w2_mm", None)


class NetworkHelper:
    """NetworkHelper for decoupling algorithm with network framework."""
    def create_network(self):
        """
        Create a network.

        Returns:
            Created network.

        Examples:
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> network = helper.create_network()
        """
        raise NotImplementedError

    def get_spec(self, name: str):
        """
        Get network specific, such as batch_size, seq_length and so on.

        Args:
            name (str): Name of specific.

        Returns:
            Object as network specific.

        Examples:
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> helper.get_spec("batch_size")
            1 (The output is related to the `mfconfig`, and the result here is just for example.)
        """
        raise NotImplementedError

    def create_tokenizer(self, **kwargs):
        """
        Get network tokenizer.

        Args:
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            Object as network tokenizer.

        Examples:
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> helper.create_tokenizer()
            LlamaTokenizer(name_or_path='', vocab_size=32000, model_max_length=100000,  added_tokens_decoder={
                0: AddedToken("<unk>", rstrip=False, lstrip=False, normalized=True, special=True),
                1: AddedToken("<s>", rstrip=False, lstrip=False, normalized=True, special=True),
                2: AddedToken("</s>", rstrip=False, lstrip=False, normalized=True, special=True),
            }
        """
        raise NotImplementedError

    def generate(self, network: Cell, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=None, **kwargs):
        """
        Invoke `network` and generate tokens.

        Args:
            network (Cell): Network to generate tokens.
            input_ids (numpy.ndarray): Input tokens for generate.
            max_new_tokens (int): Max number of tokens to be generated, default 1.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list as generated tokens.

        Examples:
            >>> import numpy as np
            >>> from mindspore import context
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers import LlamaForCausalLM, LlamaConfig
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
            >>> input_ids = np.array([[1, 10000]], dtype = np.int32)
            >>> helper.generate(network, input_ids)
            array([[    1, 10000, 10001]], dtype=int32)
        """
        raise NotImplementedError

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        """
        Assemble network inputs for predict from input tokens in numpy ndarray format.

        Args:
            input_ids (numpy.ndarray): Input tokens.
            kwargs (Dict): Extensible parameter for subclasses.

        Returns:
            A list of `mindspore.Tensor` as inputs of network predict.

        Examples:
            >>> import numpy as np
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> input_ids = np.array([[1, 10000]], dtype = np.int32)
            >>> helper.assemble_inputs(input_ids)
            (Tensor(shape=[1, 4096], dtype=Int32, value=
            [[    1, 10000,     0 ...     0,     0]]), None, None, None, None, None, None, None, None, None, \
             Tensor(shape=[1, 256], dtype=Int32, value=
            [[  0,   1,   2 ... 253, 254, 255]]), Tensor(shape=[2], dtype=Int32, value= [0, 1]))
        """
        raise NotImplementedError

    def analysis_decoder_groups(self, network):
        """
        Analyze decoder groups information of network.

        Args:
            network (Cell): network to analyze decoder groups information.

        Examples:
            >>> from mindspore import context
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers import LlamaForCausalLM, LlamaConfig
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
            >>> helper.analysis_decoder_groups(network)
        """
        raise NotImplementedError

    def get_pre_layer(self, linear_name):
        """
        Get pre layer information from current linear_name.

        Args:
            linear_name (str): linear layer name.

        Returns:
            A dict of pre layer information which include pre layer name、layer and type.

        Examples:
            >>> from mindspore import context
            >>> from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
            >>> from mindformers import LlamaForCausalLM, LlamaConfig
            >>> from mindformers.tools.register.config import MindFormerConfig
            >>> context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
            >>> mf_yaml_config_file = "/path/to/mf_yaml_config_file"
            >>> mfconfig = MindFormerConfig(mf_yaml_config_file)
            >>> helper = MFLlama2Helper(mfconfig)
            >>> network = LlamaForCausalLM(LlamaConfig(**mfconfig.model.model_config))
            >>> helper.analysis_decoder_groups(network)
            >>> helper.get_pre_layer(linear_name)
        """
        raise NotImplementedError
