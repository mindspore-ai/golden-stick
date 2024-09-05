# Copyright 2023 Huawei Technologies Co., Ltd
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
"""A network iterator for transforming network."""

import abc
import warnings
from typing import Tuple

from mindspore.nn import Cell
from mindspore_gs.common import logger


class Processor(abc.ABC):
    """A network iterator for transforming network."""
    @abc.abstractmethod
    def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
        """Callback function when visiting to each cell."""
        raise NotImplementedError

    def process(self, root: Cell, name_prefix: str = "root"):
        """Iterate the whole network and call callback function `process_cell`."""
        if root is None:
            return root
        for name, cell in root.name_cells().items():
            full_cell_name = f"{name_prefix}.{name}"
            new_cell, is_end_point = self.process_cell(full_cell_name, cell)
            if new_cell is not cell:
                root.insert_child_to_cell(name, new_cell)
            if not is_end_point:
                _ = self.process(new_cell, full_cell_name)
        return root


def transform_network_inplace(network: Cell, target_layer_type: type, transform_fn, network_root_name="root"):
    class Transformer(Processor):
        def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
            if not isinstance(cell, target_layer_type):
                return cell, False
            transform_fn(cell_name, cell)
            return cell, True

    Transformer().process(network, network_root_name)


def network_replace(network: Cell, src_layer: type, dst_layer: type, dst_layer_fn: callable, opname_blacklist: list,
                    network_root_name="root"):
    """network replace"""
    class Replacer(Processor):
        """A network iterator for one-to-one sub-cell replace."""
        def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
            if not isinstance(cell, src_layer):
                return cell, False
            if dst_layer and isinstance(cell, dst_layer):
                return cell, True
            for opname in opname_blacklist:
                if opname in cell_name:
                    logger.info(f"{cell_name} is in blacklist, keep not being replaced.")
                    return cell, True
            new_dst_layer = dst_layer_fn(cell_name, cell)
            logger.info(f"replacing {cell_name} with layer({dst_layer}).")
            nonlocal changed
            changed = True
            return new_dst_layer, True

    changed = False
    Replacer().process(network, network_root_name)
    if not changed:
        warn_str = f"Not found {src_layer} of layer to quant, please check network and opname_blacklist" \
                    f"({opname_blacklist})."
        warnings.warn(warn_str, RuntimeWarning)
