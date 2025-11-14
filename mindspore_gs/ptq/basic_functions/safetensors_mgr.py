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
"""SafeTensorsMgr"""


import os
import time
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import mindspore as ms
from mindspore.communication import get_rank, get_group_size
from mindspore_gs.common import logger
from mindspore_gs.ptq.basic_functions.distributed_parameter import DistributedParameter


class SafeTensorsMgr:
    """SafeTensorsMgr"""

    def __init__(self, file_limit_g=4):
        self._parameters = []
        self.file_limit_g = file_limit_g
        try:
            self.rank_id = get_rank()
            self.group_size = get_group_size()
        except RuntimeError:
            self.rank_id = 0
            self.group_size = 1
        if self.group_size > 1:
            self.barrier = ms.ops.Barrier()

    def save(self, original_path, save_path, dis_params_dict, quant_desc_info):
        """save"""
        start = time.time()
        if self.group_size > 1:
            self._tp_merge(dis_params_dict)
        if self.rank_id == 0:
            os.makedirs(save_path, exist_ok=True)
            index_json, inv_index_json, total_bytes = self._index_and_size(dis_params_dict)

            SafeTensorsMgr._copy_original_files(original_path, save_path)
            SafeTensorsMgr._save_sf_index_json(save_path, index_json, total_bytes)
            SafeTensorsMgr._save_quant_desc_json(save_path, quant_desc_info)
            SafeTensorsMgr._save_safetensors(save_path, inv_index_json)
            logger.info(f'Save safetensors cost time is {time.time() - start} s.')
        if self.group_size > 1:
            self.barrier()
            logger.info(f"barrier finish at rank {self.rank_id}.")

    @staticmethod
    def _copy_original_files(original_path, save_path):
        """
        Copy files from original directory to save directory with blacklist filtering.

        This method copies all files from the original directory to the save directory,
        except for files that match patterns in the blacklist. The blacklist currently
        includes files ending with '.index.json' and '.safetensors' to avoid copying
        index files and safetensors files that will be regenerated.

        Args:
            original_path (str): Path to the source directory containing original files.
            save_path (str): Path to the destination directory where files will be copied.

        Raises:
            FileNotFoundError: If the original_path does not exist.
            NotADirectoryError: If the original_path is not a directory.

        Note:
            - Path validation is performed before file operations to ensure robustness.
            - Only files in the root of original_path are processed (no subdirectories).
            - File permissions and metadata are preserved using shutil.copy2.
            - Files matching blacklist patterns are silently skipped.
        """
        src_path = Path(original_path)

        # Validate that the source path exists and is a directory
        if not src_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {original_path}")
        if not src_path.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {original_path}")

        # Define blacklist for files that should not be copied
        blacklist_patterns = [
            '.index.json',  # Original blacklist item
            '.safetensors'  # Exclude safetensors files as requested
        ]

        # Iterate through all files in the source directory
        for file_path in src_path.iterdir():
            if file_path.is_file():
                # Check if file matches any blacklist pattern
                should_skip = False
                for pattern in blacklist_patterns:
                    if file_path.name.endswith(pattern):
                        should_skip = True
                        break

                # Copy file if it's not in blacklist
                if not should_skip:
                    shutil.copy2(file_path, os.path.join(save_path, file_path.name))

    def _tp_merge(self, dis_params_dict: dict[str, DistributedParameter]):
        sorted_params = sorted(dis_params_dict)
        enable_tqdm = self.rank_id == 0
        for name in tqdm(sorted_params, desc="Merge TP weights", disable=not enable_tqdm):
            param = dis_params_dict[name]
            param.comm()
            self.barrier()

    @staticmethod
    def _get_num_str(index, length=5):
        if index < 0:
            raise RuntimeError(f"index should be bigger than 0, but got {index}.")
        for i in range(length):
            threshold = 10 ** (i + 1)
            if index < threshold:
                return f"{'0' * (length - i - 1)}{index}"
        raise RuntimeError(f"index should be smaller than {10 ** length}, but got {index}.")

    @staticmethod
    def _get_sf_file_name(cur_index, total_num_str):
        return f"quant-model-{SafeTensorsMgr._get_num_str(cur_index)}-of-{total_num_str}.safetensors"

    def _index_and_size(self, dis_params_dict: dict[str, DistributedParameter]):
        """get index and size from dis_params_dict"""
        total_bytes = 0
        cur_bytes = 0
        cur_index = 1
        index_json = {}
        inv_index_json: dict[str, dict] = {}
        cur_params = {}
        limits = 1024 * 1024 * 1024 * self.file_limit_g
        for name, param in dis_params_dict.items():
            param_size = param.size()
            total_bytes += param_size
            cur_bytes += param_size
            if cur_bytes > limits:
                inv_index_json[cur_index] = cur_params.copy()
                cur_params.clear()
                cur_index += 1
                cur_bytes = param_size
            index_json[name] = cur_index
            cur_params[name] = param.param
        inv_index_json[cur_index] = cur_params.copy()
        total_num_str = SafeTensorsMgr._get_num_str(cur_index)
        new_index_json = {}
        for name, index in index_json.items():
            new_index_json[name] = SafeTensorsMgr._get_sf_file_name(index, total_num_str)
        new_inv_index_json = {}
        for index, cur_params in inv_index_json.items():
            new_inv_index_json[SafeTensorsMgr._get_sf_file_name(index, total_num_str)] = cur_params
        return new_index_json, new_inv_index_json, total_bytes

    @staticmethod
    def _save_quant_desc_json(save_path, quant_desc):
        """_save_desc_json"""
        save_json_path = os.path.join(save_path, "quantization_description.json")
        os.makedirs(save_path, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(quant_desc, f, ensure_ascii=False, indent=4)
        logger.info(f'Quantization describle json file saved to {save_json_path}', flush=True)

    @staticmethod
    def _save_sf_index_json(save_path, index_json, total_bytes):
        """_save_desc_json"""
        save_json_path = os.path.join(save_path, "model.safetensors.index.json")
        os.makedirs(save_path, exist_ok=True)
        index_data = {
            "metadata": {"total_size": total_bytes},
            "weight_map": index_json
        }
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=4)
        logger.info(f'SafeTensors index json file saved to {save_json_path}', flush=True)

    @staticmethod
    def _save_safetensors(save_path, inv_index_json):
        for sf_file, cur_params in inv_index_json.items():
            sf_path = os.path.join(save_path, sf_file)
            logger.debug(f"Save {cur_params.keys()} to {sf_path}")
            ms.save_checkpoint(cur_params, sf_path, format="safetensors")
        logger.debug(f"SafeTensors saved to {save_path}")
