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

__global__ void UpdateInputMinMaxPerChannel(float *input, float *input_min, float *input_max, float *output_min,
                                            float *output_max, int channels, int per_channel_nums, bool ema,
                                            float ema_decay) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channels; i += blockDim.x * gridDim.x) {
    thrust::pair<float *, float *> sum =
      thrust::minmax_element(thrust::device, input + i * per_channel_nums, input + per_channel_nums * (i + 1));
    if (ema) {
      output_min[i] = ema_decay * sum.first[0] + (1 - ema_decay) * input_min[i];
      output_max[i] = ema_decay * sum.second[0] + (1 - ema_decay) * input_max[i];
    } else {
      output_min[i] = sum.first[0];
      output_max[i] = sum.second[0];
    }
    output_min[i] = output_min[i] > 0 ? 0 : output_min[i];
    output_max[i] = output_max[i] < 0 ? 0 : output_max[i];
  }
  return;
}

void CalMinMaxPerChannel(float *input, float *input_min, float *input_max, float *output_min, float *output_max,
                         const int total_num, const int channel_num, const float ema_decay, const bool ema,
                         cudaStream_t cuda_stream) {
  int per_channel_num = total_num / channel_num;
  UpdateInputMinMaxPerChannel<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(
    input, input_min, input_max, output_min, output_max, channel_num, per_channel_num, ema, ema_decay);
  return;
}

class minmax_update_perchannel_kernel_attr : public AotKernelData {
 public:
  float ema_decay;
  bool ema;
};

extern "C" int MinmaxUpdatePerChannelInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  minmax_update_perchannel_kernel_attr *kernel_ptr = new minmax_update_perchannel_kernel_attr;
  kernel_ptr->ema = extra->Attr<bool>("ema");
  kernel_ptr->ema_decay = extra->Attr<float>("ema_decay");
  extra->SetKernelData(kernel_ptr);

  return 0;
}

extern "C" int MinmaxUpdatePerChannel(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                                      void *stream, void *extra_void) {
  constexpr int TOTAL_PARAM_NUM = 3 + 2;  // input 3, output 2
  constexpr int INPUT_INDEX = 0;
  constexpr int IO_NUM = 5;

  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  for (int index = 0; index < IO_NUM; index++) {
    if (strcmp(dtypes[index], "float32") != 0) {
      return 2;
    }
  }

  float *input_data = static_cast<float *>(params[0]);
  float *input_min = static_cast<float *>(params[1]);
  float *input_max = static_cast<float *>(params[2]);
  float *output_min = static_cast<float *>(params[3]);
  float *output_max = static_cast<float *>(params[4]);

  int size = 1;
  for (int i = 0; i < ndims[INPUT_INDEX]; i++) {
    size *= shapes[INPUT_INDEX][i];
  }
  size_t num_channels = static_cast<size_t>(shapes[0][0]);

  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<minmax_update_perchannel_kernel_attr *>(extra->KernelData());
  float ema_decay = kernel_ptr->ema_decay;
  bool ema = kernel_ptr->ema;

  CalMinMaxPerChannel(input_data, input_min, input_max, output_min, output_max, size, num_channels, ema_decay, ema,
                      reinterpret_cast<cudaStream_t>(stream));

  return 0;
}
