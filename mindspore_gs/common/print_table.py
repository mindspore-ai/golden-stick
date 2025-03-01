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
""" print table """

from prettytable import PrettyTable


def print_table(title, desc_name, infos):
    """print_table"""
    x = PrettyTable()
    x.title = title
    x.field_names = ['layer_name', desc_name]
    for info in infos:
        x.add_row(info)
    print(x)


if __name__ == "__main__":
    items = [('network.model.layers.0.attention.wq', 'A8W8'),
             ('network.model.layers.0.attention.wk', 'A8W8'),
             ('network.model.layers.0.attention.wv', 'A8W8'),
             ('network.model.layers.0.attention.wo', 'A8W8'),
             ('network.model.layers.0.feed_forward.w1', 'A8pertoken-W8perchannel'),
             ('network.model.layers.0.feed_forward.w3', 'A8pertoken-W8perchannel'),
             ('network.model.layers.0.feed_forward.w2', 'not quant'),
            ]
    print_table('Network Quantization Summary', 'quant_type', items)
