#!/bin/bash
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

echo "---------------- GoldenStick: build start ----------------"
BASEPATH=$(cd "$(dirname $0)"; pwd)

# Get target backend from command line argument or environment variable, default to ascend
TARGET=${1:-${GS_TARGET:-ascend}}
if [ "${TARGET}" != "ascend" ] && [ "${TARGET}" != "gpu" ]; then
    echo "Warning: Invalid target '${TARGET}', defaulting to 'ascend'"
    TARGET="ascend"
fi

echo "Building for target: ${TARGET}"
python3 setup.py bdist_wheel -d ${BASEPATH}/output --target=${TARGET}

if [ ! -d "${BASEPATH}/output" ]; then
    echo "The directory ${BASEPATH}/output dose not exist."
    exit 1
fi

rm -rf "mindspore_gs.egg-info"

cd ${BASEPATH}/output || exit
for package in mindspore_gs*whl
do
    [[ -e "${package}" ]] || break
    sha256sum ${package} > ${package}.sha256
done
cd ${BASEPATH} || exit
echo "---------------- GoldenStick: build end   ----------------"
