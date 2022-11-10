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

__global__ void UpdateInputMinMaxPerLayerWithEMA(const float *input_min, const float *input_max, float *output_min,
                                                 float *output_max, const float min, const float max,
                                                 const float decay) {
  output_min[0] = decay * (min) + (1 - decay) * (input_min[0]);
  output_min[0] = output_min[0] > 0 ? 0 : output_min[0];
  output_max[0] = decay * (max) + (1 - decay) * (input_max[0]);
  output_max[0] = output_max[0] < 0 ? 0 : output_max[0];
  return;
}

__global__ void UpdateInputMinMaxPerLayer(float *output_min, float *output_max, const float min, const float max) {
  output_min[0] = min > 0 ? 0 : min;
  output_max[0] = max < 0 ? 0 : max;
  return;
}

void CalMinMaxPerLayer(float *input, float *input_min, float *input_max, float *output_min, float *output_max,
                       const int total_num, const float ema_decay, const bool ema, cudaStream_t cuda_stream) {
  float minel = 0.f;
  float maxel = 0.f;
  auto policy = thrust::cuda::par.on(cuda_stream);
  thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> tuple;
  tuple =
    thrust::minmax_element(policy, thrust::device_pointer_cast(input), thrust::device_pointer_cast(input) + total_num);
  minel = tuple.first[0];
  maxel = tuple.second[0];

  if (ema) {
    UpdateInputMinMaxPerLayerWithEMA<<<1, 1, 0, cuda_stream>>>(input_min, input_max, output_min, output_max, minel,
                                                               maxel, ema_decay);
  } else {
    UpdateInputMinMaxPerLayer<<<1, 1, 0, cuda_stream>>>(output_min, output_max, minel, maxel);
  }
  return;
}

class minmax_update_perlayer_kernel_attr : public AotKernelData {
 public:
  float ema_decay;
  bool ema;
};

extern "C" int CustomMinMaxUpdatePerLayerInit(int *ndims, int64_t **shapes, const char **dtypes, AotExtra *extra) {
  minmax_update_perlayer_kernel_attr *kernel_ptr = new minmax_update_perlayer_kernel_attr;
  kernel_ptr->ema = extra->Attr<bool>("ema");
  kernel_ptr->ema_decay = extra->Attr<float>("ema_decay");
  extra->SetKernelData(kernel_ptr);

  return 0;
}

extern "C" int CustomMinMaxUpdatePerLayer(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
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

  AotExtra *extra = static_cast<AotExtra *>(extra_void);
  auto kernel_ptr = static_cast<minmax_update_perlayer_kernel_attr *>(extra->KernelData());
  float ema_decay = kernel_ptr->ema_decay;
  bool ema = kernel_ptr->ema;

  CalMinMaxPerLayer(input_data, input_min, input_max, output_min, output_max, size, ema_decay, ema,
                    reinterpret_cast<cudaStream_t>(stream));

  return 0;
}
