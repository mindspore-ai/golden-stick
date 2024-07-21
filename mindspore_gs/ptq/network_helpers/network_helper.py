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


import numpy as np
from mindspore.nn import Cell


class NetworkHelper:
    """NetworkHelper for decoupling algorithm with network framework."""
    def get_spec(self, name: str):
        """
        Get network specific, such as batch_size, seq_length and so on.

        Args:
            name (str): Name of specific.

        Returns:
            Object as network specific.

        Examples:
            >>> from mindspore_gs.ptq import MFLlama2Helper
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
            >>> from mindspore_gs.ptq import MFLlama2Helper
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

    def generate(self, network: Cell, input_ids: np.ndarray, max_new_tokens=1, **kwargs):
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
            >>> from mindspore_gs.ptq import MFLlama2Helper
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
            >>> from mindspore_gs.ptq import MFLlama2Helper
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
