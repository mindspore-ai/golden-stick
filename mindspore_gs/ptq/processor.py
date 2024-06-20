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
from typing import Tuple

from mindspore.nn import Cell


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
            new_cell, is_end_point = self.process_cell(name, cell)
            if new_cell is not cell:
                root.insert_child_to_cell(name, new_cell)
            if not is_end_point:
                _ = self.process(new_cell)
        return root
