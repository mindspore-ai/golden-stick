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

#include "fake_quant_impl.cuh"

__global__ void FQPerChannelGrad(const float *input, const float *gradient, float *output, const int total_size,
                                 const int channel_size, const float *nudge_min, const float *nudge_max) {
  int channel_idx = 0;
  int per_channel_num = total_size / channel_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
    channel_idx = floor(static_cast<double>(i) / static_cast<double>(per_channel_num));
    if (input[i] < nudge_min[channel_idx] || input[i] > nudge_max[channel_idx]) {
      output[i] = 0;
    } else {
      output[i] = gradient[i];
    }
  }
}

void CalFQPerChannelGrad(const float *input, const float *gradient, float *output, const int total_num,
                         const int channel_num, const float *nudge_min, const float *nudge_max,
                         cudaStream_t cuda_stream) {
  FQPerChannelGrad<<<GET_BLOCKS(channel_num), GET_THREADS, 0, cuda_stream>>>(input, gradient, output, total_num,
                                                                             channel_num, nudge_min, nudge_max);
}

class FQPerChannelGradKernelAttr : public AotKernelData {
 public:
  int num_bits;
  bool symmetric;
  bool narrow_range;
  int quant_delay;
};

extern "C" int CustomFakeQuantPerChannelGradInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  size_t num_channels = static_cast<size_t>(shapes[0][0]);
  extra->SetWorkSpace({num_channels * sizeof(float), num_channels * sizeof(float), num_channels * sizeof(float)});

  FQPerChannelGradKernelAttr *kernel_ptr = new FQPerChannelGradKernelAttr;
  kernel_ptr->num_bits = static_cast<int>(extra->Attr<int64_t>("num_bits"));
  kernel_ptr->symmetric = extra->Attr<bool>("symmetric");
  kernel_ptr->narrow_range = extra->Attr<bool>("narrow_range");
  kernel_ptr->quant_delay = static_cast<int>(extra->Attr<int64_t>("quant_delay"));
  extra->SetKernelData(kernel_ptr);

  return 0;
}

int global_step = 0;

extern "C" int CustomFakeQuantPerChannelGrad(int nparam, void **params, int *ndims, int64_t **shapes,
                                             const char **dtypes, void *stream, void *extra_void) {
  constexpr int TOTAL_PARAM_NUM = 4 + 1 + 3;  // input 4, output 1, workspace 3
  constexpr int IO_NUM = 5;
  constexpr int OUTPUT_INDEX = 4;

  if (nparam != TOTAL_PARAM_NUM) {
    return 1;
  }

  for (int index = 0; index < IO_NUM; index++) {
    if (strcmp(dtypes[index], "float32") != 0) {
      return 2;
    }
  }

  float *gradient = static_cast<float *>(params[0]);
  float *input_data = static_cast<float *>(params[1]);
  float *input_min = static_cast<float *>(params[2]);
  float *input_max = static_cast<float *>(params[3]);
  float *output = static_cast<float *>(params[4]);
  float *w_scale = static_cast<float *>(params[5]);
  float *w_nudge_min = static_cast<float *>(params[6]);
  float *w_nudge_max = static_cast<float *>(params[7]);

  int size = 1;
  for (int i = 0; i < ndims[OUTPUT_INDEX]; i++) {
    size *= shapes[OUTPUT_INDEX][i];
  }

  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<FQPerChannelGradKernelAttr *>(extra->KernelData());
  int num_bits = kernel_ptr->num_bits;
  if (num_bits <= 2 || num_bits >= 16) {
    return 3;
  }
  bool symmetric = kernel_ptr->symmetric;
  bool narrow_range = kernel_ptr->narrow_range;
  int quant_delay = kernel_ptr->quant_delay;
  if (quant_delay < 0) {
    return 3;
  }
  float quant_min = 0;
  float quant_max = (1 << num_bits) - 1;
  if (narrow_range) {
    quant_min++;
  }
  size_t num_channels = static_cast<size_t>(shapes[0][0]);

  if (global_step >= quant_delay) {
    CalNudgePerChannel(input_min, input_max, quant_min, quant_max, w_nudge_min, w_nudge_max, w_scale, num_channels,
                       symmetric, reinterpret_cast<cudaStream_t>(stream));
    CalFQPerChannelGrad(input_data, gradient, output, size, num_channels, w_nudge_min, w_nudge_max,
                        reinterpret_cast<cudaStream_t>(stream));
  } else {
    CHECK_CUDA_RET_WITH_ERROR("FakeQuantPerChannelGrad",
                              cudaMemcpyAsync(output, gradient, size * sizeof(float), cudaMemcpyDeviceToDevice,
                                              reinterpret_cast<cudaStream_t>(stream)),
                              "Copy gpu failed.");
  }
  global_step++;

  return 0;
}
