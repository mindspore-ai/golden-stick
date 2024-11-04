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
"""Define version of MindSpore Golden Stick and check version matching between MindSpore and MindSpore Golden Stick."""

import time

#pylint: disable=C0111
__version__ = '0.6.1'


def mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Golden Stick. If the MindSpore can not be imported, it will raise
    an ImportError. If its version is not compatible with current MindSpore Golden Stick version, it will print a
    warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except (ImportError, ModuleNotFoundError):
        print("Can not find MindSpore in current environment. Please install MindSpore before using MindSpore Golden "
              "Stick, by following the instruction at https://www.mindspore.cn/install")
        raise

    ms_msgs_version_match = {'0.1': '1.8',
                             '0.2': '1.9',
                             '0.3': '2.0',
                             '0.4': '2.3',
                             '0.5': '2.3',
                             '0.6': '2.4'}

    required_mindspore_verision = ms_msgs_version_match[__version__[:3]]
    ms_version = ms.__version__[:3]

    if required_mindspore_verision != ms_version:
        logger.warning("Current version of MindSpore is not compatible with MindSpore Golden Stick. Some functions "
                       "might not work or even raise error. Please install MindSpore version == {}. For more details "
                       "about dependency setting, please check the instructions at MindSpore official website "
                       "https://www.mindspore.cn/install or check the README.md at "
                       "https://gitee.com/mindspore/golden-stick".format(required_mindspore_verision))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(f"Please pay attention to the above warning, countdonw: {i}")
            time.sleep(1)
