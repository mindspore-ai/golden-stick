#!/bin/bash
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

if [ $# -ne 1 ]; then
  script_name=$(basename "$0")
  echo "usage: bash ${script_name} version, available version: master, r0.6.0, r0.5.0, r0.2.0; etc"
  exit 1
fi

version=$1

CUR_DIR=$(cd "$(dirname $0)"; pwd)

# mindspore_gs/pruner/README.md
sed -i "s/\.\/scop\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/pruner\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/pruner/README_CN.md
sed -i "s/\.\/scop\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/pruner\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/pruner/README.md

# mindspore_gs/ptq/README.md
sed -i "s/\.\.\/quantization\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md
sed -i "s/\.\/ptq\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md
sed -i "s/\.\/round_to_nearest\/README_CN\.ipynb/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/round_to_nearest\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README_CN.md

sed -i "s/\.\.\/quantization\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md
sed -i "s/\.\/ptq\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md
sed -i "s/\.\/round_to_nearest\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/round_to_nearest\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/ptq/README.md

# mindspore_gs/quantization/README.md
sed -i "s/\.\.\/simulated_quantization\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/simqat\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md
sed -i "s/\.\.\/slb\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/quantization\/slb\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md
sed -i "s/\.\.\/ptq\/README_CN\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/zh-CN\/${version}\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README_CN.md

sed -i "s/\.\.\/simulated_quantization\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/simqat\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md
sed -i "s/\.\.\/slb\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/quantization\/slb\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md
sed -i "s/\.\.\/ptq\/README\.md/https:\/\/www\.mindspore\.cn\/golden_stick\/docs\/en\/${version}\/ptq\/overview\.html/g" ${CUR_DIR}/../mindspore_gs/quantization/README.md
