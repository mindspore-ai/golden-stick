#!/bin/bash
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

# usage: bash ci-test.sh cpu/gpu
CURRPATH=$(cd "$(dirname $0)" || exit; pwd)

backend=${1-'cpu'}
sub_dir=${2-'st'}

cd ${CURRPATH} || exit 1
if [[ ${backend} == 'cpu' ]]; then
  # shellcheck disable=SC2038
  find ${sub_dir} -name 'test_*.py' -type f|xargs -r grep -A 6 'pytest.mark.level0'|grep 'pytest.mark.platform_x86_cpu'|awk -F'-' '{print $1}'|uniq|xargs python -m pytest -vrt -q -m 'level0 and platform_x86_cpu'
elif [[ ${backend} == 'gpu' ]]; then
  echo "Please ensure dataset path is exported to env. for example, cifar10 is installed in: /path/to/ds/cifar/cifar-10-batches-bin/xxx.bin, and mnist is installed in: /path/to/ds/mnist/train/xxx-ubyte, you should export DATASET_PATH=/path/to/ds."
  # shellcheck disable=SC2038
  find ${sub_dir} -name 'test_*.py' -type f|xargs -r grep -A 6 'pytest.mark.level0'|grep 'pytest.mark.platform_x86_gpu_training'|awk -F'-' '{print $1}'|uniq|xargs python -m pytest -vrt -q -m 'level0 and platform_x86_gpu_training'
elif [[ ${backend} == 'ascend' ]]; then
  # shellcheck disable=SC2038
  find ${sub_dir} -name 'test_*.py' -type f|xargs -r grep -A 6 'pytest.mark.level0'|grep 'pytest.mark.platform_arm_ascend910b_training'|awk -F'-' '{print $1}'|uniq|xargs python -m pytest -vrt -q -m 'level0 and platform_arm_ascend910b_training'
else
  echo "There is no ${backend} backend testcases, available: cpu, gpu, ascend."
fi
cd - || exit 1
exit 0
