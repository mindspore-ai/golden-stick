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
"""
util functions for golden-stick
"""

import types
import warnings
from functools import wraps
import psutil
import numpy as np
from mindspore import nn


def value_check(name, src, supported_type, value_choices=None):
    """Check if the given object is the given supported type and in the given supported value.

    Example::
        >>> from .utils import value_check
        >>> value_check('my_var', my_var, str, ['my', 'var'])
    """
    if not isinstance(src, supported_type):
        raise TypeError("Type of {} should be {} but got {}".format(name, str(supported_type), type(src)))

    if value_choices is not None and value_choices:
        if src not in value_choices:
            raise ValueError("{} is not in supported {}: {}. Skip setting it.".format(src, name, str(value_choices)))


def list_value_check(name, src, item_supported_type, value_choices=None):
    """Check if the given list object is the given supported type and in the given supported value.
    """
    if not isinstance(src, (list, tuple)):
        raise TypeError("Type of {} should be a list but got {}".format(name, type(src)))
    if any([not isinstance(i, item_supported_type) for i in src]):
        raise TypeError("Type of item of {} should be one of {} but got {}".format(name, str(item_supported_type),
                                                                                   [type(i) for i in src]))
    if value_choices is not None and not value_choices:
        if isinstance(src, str) and src not in value_choices:
            raise ValueError("{} is not in supported {}: {}. Skip setting it.".format(src, name, str(value_choices)))
        if isinstance(src, list) and all([isinstance(i, str) for i in src])\
                and any([i not in value_choices for i in src]):
            raise ValueError("{} is not in supported {}: {}. Skip setting it.".format(src, name, str(value_choices)))


def offload_network(network: nn.Cell):
    """offload_network"""
    for _, param in network.parameters_dict().items():
        # pylint: disable=protected-access
        param._offload()


def deprecated(version, substitute=None):
    """deprecated warning

    Args:
        version (str): version that the operator or function is deprecated.
        substitute (str): the substitute name for deprecated operator or function.
    """

    def decorate(target):
        warnings.filterwarnings("always", category=DeprecationWarning)
        warn_string = (
            f"'{target.__name__}' is deprecated from version {version} and "
            f"will be removed in a future version."
        )
        subsequent_indent = "    "
        doc_string = f"{subsequent_indent}.. deprecated:: {version}"
        target_type = "method"
        if isinstance(target, type):
            target_type = "class"
        if isinstance(target, types.FunctionType):
            target_type = "function"

        if substitute:
            doc_string = (
                f"{doc_string}\n{subsequent_indent}{subsequent_indent}"
                f"Use :{target_type}:`{substitute}` instead."
            )
            warn_string = f"{warn_string} Use '{substitute}' instead."

        target.__doc__ = f"{doc_string}\n\n{target.__doc__}"
        if isinstance(target, type):
            origin_init = target.__init__

            @wraps(origin_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(warn_string, category=DeprecationWarning)
                return origin_init(self, *args, **kwargs)

            target.__init__ = new_init
            return target

        if callable(target):

            @wraps(target)
            def wrapper(*args, **kwargs):
                warnings.warn(warn_string, category=DeprecationWarning)
                ret = target(*args, **kwargs)
                return ret

            return wrapper
        return target

    return decorate


def check_nan_inf(arr: np.ndarray):
    """check_nan_inf"""
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))
    return has_nan, has_inf


def get_memory_info():
    """get_memory_info"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.2f} MB") # 物理内存
    print(f"VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.2f} MB") # 虚拟内存
