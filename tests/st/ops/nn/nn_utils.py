# Copyright 2022 Huawei Technologies Co., Ltd
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
"""utils for nn cell."""
from mindspore_gs.ops.nn import FakeQuantWithMinMaxObserver
from mindspore_gs.ops.nn.fake_quant_with_min_max_observer import QuantConfig
from mindspore.compression.common import QuantDtype


def create_quant_config(quant_observer=(FakeQuantWithMinMaxObserver, FakeQuantWithMinMaxObserver),
                        quant_delay=(0, 0),
                        quant_dtype=(QuantDtype.INT8, QuantDtype.INT8),
                        per_channel=(False, False),
                        symmetric=(False, False),
                        narrow_range=(False, False),
                        mode="DEFAULT"):
    """
    Config the observer type of weights and data flow with quant parameters.
    """
    if per_channel[-1]:
        raise ValueError("Arg 'per_channel' second element must be 'False'.")
    weight_observer = quant_observer[0].partial_init(quant_delay=quant_delay[0], quant_dtype=quant_dtype[0],
                                                     per_channel=per_channel[0], symmetric=symmetric[0],
                                                     narrow_range=narrow_range[0], mode=mode)
    act_observer = quant_observer[-1].partial_init(quant_delay=quant_delay[-1], quant_dtype=quant_dtype[-1],
                                                   per_channel=per_channel[-1], symmetric=symmetric[-1],
                                                   narrow_range=narrow_range[-1], mode=mode)
    return QuantConfig(weight=weight_observer, activation=act_observer)
