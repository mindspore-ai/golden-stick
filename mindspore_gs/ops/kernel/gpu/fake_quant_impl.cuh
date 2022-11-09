/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include "aot/custom_aot_extra.h"

#define GET_THREADS 1024
#define GET_BLOCKS(channel_num) channel_num / GET_THREADS + 1
#define CHECK_CUDA_RET_WITH_ERROR(ops_name, expression, message)                                                   \
  {                                                                                                                \
    cudaError_t status = (expression);                                                                             \
    if (status != cudaSuccess) {                                                                                   \
      std::cout << "CUDA Error: " << message << " | Error Number: " << status << " " << cudaGetErrorString(status) \
                << "in ops " << ops_name << "\n";                                                                  \
    }                                                                                                              \
  }

/**
 * Find the nudge min, max and scale value as output.
 * @param input_min array
 * @param input_max array
 * @param quant_min 1 << bit -1
 * @param quant_max 0
 * @param nudge_min array
 * @param nudge_max array
 * @param scale array
 * @param channel_num
 * @return
 */
__global__ void NudgeMinMaxPerChannel(float *input_min, float *input_max, const float quant_min, const float quant_max,
                                      float *nudge_min, float *nudge_max, float *scale, int channel_num,
                                      const bool symmetric) {
  float zp_from_min = 0.f;
  float nudge_zp = 0.f;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channel_num; i += blockDim.x * gridDim.x) {
    if (symmetric) {
      input_max[i] = abs(input_min[i]) < input_max[i] ? input_max[i] : -input_min[i];
      input_min[i] = abs(input_min[i]) < input_max[i] ? -input_max[i] : input_min[i];
    }
    if ((quant_max - quant_min) == 0 || (input_max[i] - input_min[i]) == 0) {
      scale[i] = 0.f;
      zp_from_min = 0.f;
    } else {
      scale[i] = (input_max[i] - input_min[i]) / (quant_max - quant_min);
      zp_from_min = quant_min - input_min[i] / scale[i];
    }

    if (zp_from_min <= quant_min) {
      nudge_zp = quant_min;
    } else if (zp_from_min >= quant_max) {
      nudge_zp = quant_max;
    } else {
      nudge_zp = round(zp_from_min);
    }

    nudge_min[i] = (quant_min - nudge_zp) * (scale[i]);
    nudge_max[i] = (quant_max - nudge_zp) * (scale[i]);
  }
}

void CalNudgePerChannel(float *input_min, float *input_max, const float quant_min, const float quant_max,
                        float *nudge_min, float *nudge_max, float *scale, const int channel_num, const bool symmetric,
                        cudaStream_t cuda_stream) {
  NudgeMinMaxPerChannel<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(
    input_min, input_max, quant_min, quant_max, nudge_min, nudge_max, scale, channel_num, symmetric);
}

__global__ void NudgeMinMaxPerLayer(float *input_min, float *input_max, const float quant_min, const float quant_max,
                                    float *nudge_min, float *nudge_max, float *scale, const bool symmetric) {
  float zp_from_min = 0.f;
  scale[0] = 0.f;
  nudge_max[0] = 0.f;
  nudge_min[0] = 0.f;

  if (symmetric) {
    input_max[0] = abs(input_min[0]) < input_max[0] ? input_max[0] : -input_min[0];
    input_min[0] = abs(input_min[0]) < input_max[0] ? -input_max[0] : input_min[0];
  }

  if ((quant_max - quant_min) == 0 || (input_max[0] - input_min[0]) == 0) {
    scale[0] = 0.f;
    zp_from_min = 0.f;
  } else {
    scale[0] = (input_max[0] - input_min[0]) / (quant_max - quant_min);
    zp_from_min = quant_min - input_min[0] / scale[0];
  }

  float nudge_zp = 0.f;
  if (zp_from_min <= quant_min) {
    nudge_zp = quant_min;
  } else if (zp_from_min >= quant_max) {
    nudge_zp = quant_max;
  } else {
    nudge_zp = round(zp_from_min);
  }

  nudge_min[0] = (quant_min - nudge_zp) * (scale[0]);
  nudge_max[0] = (quant_max - nudge_zp) * (scale[0]);
  return;
}

void CalNudgePerLayer(float *input_min, float *input_max, const float quant_min, const float quant_max,
                      float *nudge_min, float *nudge_max, float *scale, const bool symmetric,
                      cudaStream_t cuda_stream) {
  NudgeMinMaxPerLayer<<<1, 1, 0, cuda_stream>>>(input_min, input_max, quant_min, quant_max, nudge_min, nudge_max, scale,
                                                symmetric);
  return;
}
