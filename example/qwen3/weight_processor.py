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
transform huggingface safetensor.
"""

import os
from safetensors import safe_open
from mindspore.communication.management import get_rank, get_group_size


class BaseWeightProcessor:
    r"""
    Provide model weight load and shards.
    Args:
        config (MF Config): The config of Infer model.
        network (InferenceModelForCausalLM): The network of infer model.

    """

    def __init__(self, config, network, is_quant):
        self.config = config
        self.network = network
        self.is_quant = is_quant
        self.tp_group_size = get_group_size()
        self.rank_id = get_rank()
        self.parameter_dict = {}
        self.file_handles = {}

    def get_file_handles(self, filename):
        """get_file_handles"""
        # Validate input parameter
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")

        # Normalize and validate filename
        filename = os.path.normpath(os.path.abspath(filename))

        # Check for path traversal
        if ".." in filename:
            if not os.path.commonpath([filename, os.getcwd()]).startswith(os.getcwd()):
                raise ValueError("Invalid filename: potential path traversal detected")

        # Verify file exists and is a file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File does not exist: {filename}")
        if not os.path.isfile(filename):
            raise ValueError(f"Path is not a file: {filename}")

        # Validate file extension
        if not filename.endswith('.safetensors'):
            raise ValueError(f"Invalid file type: {filename}. Expected .safetensors file")

        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def release_file_handles(self):
        """release_file_handles"""
        del self.file_handles

    def _validate_input_parameters(self, hf_param_name, src_hf_dir, hf_weight_map):
        """Validate input parameters."""
        if not isinstance(hf_param_name, str) or not hf_param_name.strip():
            raise ValueError("hf_param_name must be a non-empty string")
        if not isinstance(src_hf_dir, str) or not src_hf_dir.strip():
            raise ValueError("src_hf_dir must be a non-empty string")
        if not isinstance(hf_weight_map, dict):
            raise ValueError("hf_weight_map must be a dictionary")

    def _validate_and_normalize_src_dir(self, src_hf_dir):
        """Validate and normalize source directory."""
        src_hf_dir = os.path.normpath(os.path.abspath(src_hf_dir))

        # Check for path traversal in src_hf_dir
        if ".." in src_hf_dir:
            if not os.path.commonpath([src_hf_dir, os.getcwd()]).startswith(os.getcwd()):
                raise ValueError("Invalid src_hf_dir: potential path traversal detected")

        # Verify src_hf_dir exists and is a directory
        if not os.path.exists(src_hf_dir):
            raise FileNotFoundError(f"Directory does not exist: {src_hf_dir}")
        if not os.path.isdir(src_hf_dir):
            raise ValueError(f"Path is not a directory: {src_hf_dir}")
        return src_hf_dir

    def _get_and_validate_safetensor_file(self, hf_param_name, hf_weight_map, src_hf_dir):
        """Get and validate safetensor file path."""
        if hf_param_name not in hf_weight_map:
            raise KeyError(f"Parameter {hf_param_name} not found in weight map")

        safetensor_file = hf_weight_map[hf_param_name]
        if not isinstance(safetensor_file, str) or not safetensor_file.strip():
            raise ValueError("safetensor_file must be a non-empty string")

        # Sanitize safetensor_file
        if ".." in safetensor_file or "/" in safetensor_file or "\\" in safetensor_file:
            raise ValueError("Invalid safetensor_file: contains path traversal characters")

        # Construct and validate filename
        filename = os.path.normpath(os.path.join(src_hf_dir, safetensor_file))

        # Ensure the filename is within the source directory
        if not filename.startswith(src_hf_dir + os.sep) and filename != src_hf_dir:
            raise ValueError("Safetensor file path outside source directory")

        # Verify file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Safetensor file does not exist: {filename}")
        if not os.path.isfile(filename):
            raise ValueError(f"Path is not a file: {filename}")

        return filename

    def _split_tensor_data(self, np_data, split_axis):
        """Split tensor data based on axis."""
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            return np_data[start:stop]
        if split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            return np_data[:, start:stop]
        if split_axis == 2:
            split_size = shape[2] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            return np_data[:, :, start:stop]
        raise ValueError("split_axis:{} is not supported.".format(split_axis))

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        """get_safetensor_from_file"""
        self._validate_input_parameters(hf_param_name, src_hf_dir, hf_weight_map)
        src_hf_dir = self._validate_and_normalize_src_dir(src_hf_dir)
        filename = self._get_and_validate_safetensor_file(hf_param_name, hf_weight_map, src_hf_dir)

        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True

        if not is_split_param:
            np_data = sf_file.get_tensor(hf_param_name)
            return np_data, qint4

        np_data = sf_file.get_slice(hf_param_name)
        split_data = self._split_tensor_data(np_data, split_axis)
        return split_data, qint4

    def split_weight_by_rank(self, weight, split_axis=0):
        """split_weight_by_rank"""
        shape = weight.shape
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[:, start:stop]
        else:
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
        return split_data

    def load_safetensors_shard(self, src_hf_dir):
        """ load safetensors and shards """
        raise NotImplementedError("load_safetensors_shard method is not implemented.")
