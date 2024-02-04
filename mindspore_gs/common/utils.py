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


def value_check(name, src, supported_type, value_choices=None):
    """Check if the given object is the given supported type and in the given supported value.

    Example::
        >>> from .utils import value_check
        >>> value_check('my_var', my_var, str, ['my', 'var'])
    """
    if isinstance(src, list) and any([not isinstance(i, supported_type) for i in src]):
        raise ValueError("Type of {} items should be {} but not {}".format(
            name, str(supported_type), [type(i) for i in src]))
    if not isinstance(src, list) and not isinstance(src, supported_type):
        raise ValueError("Type of {} should be {} but not {}".format(name, str(supported_type), type(src)))

    if value_choices is not None and not value_choices:
        if isinstance(src, str) and src not in value_choices:
            raise ValueError("{} is not in supported {}: {}. Skip setting it.".format(src, name, str(value_choices)))
        if isinstance(src, list) and all([isinstance(i, str) for i in src])\
                and any([i not in value_choices for i in src]):
            raise ValueError("{} is not in supported {}: {}. Skip setting it.".format(src, name, str(value_choices)))
