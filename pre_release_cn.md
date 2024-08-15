
## MindSpore Golden Stick 0.6.0 Release Notes

### 主要特性和增强

* `RoundToNearest` 支持将MindFormers的kvcache即 `PagedAttentionMgr` 类量化成int8，主要针对llama2系列网络。
* 新增针对MindFormers Linear层的A8W8训练后量化算法 `SmoothQuant`， 主要针对Llama2系列网络。
* 新增支持基于动态图的AutoClip和AutoScale训练后量化算法 `OmniQuant`，主要针对llama2系列网络。用户可通过配置Clip和Scale的相关超参为list或float类型，来区分是否进行参数搜索或是网络量化。
* `RoundToNearest` 新增 `load_mindformers_plugin` 静态方法，实现该算法与MindFormers的解耦。如果用户需要量化来自MindFormers的网络，需要在创建算法前调用该方法。

### API Change

* `PTQConfig`新增如下三个参数:
    * `act_quant_dtype`：mindspore.dtype类型，默认为None，可选输入及含义如下：

    |  act_quant_dtype  | mindspore.dtype.int8  | None（默认）  |
    |  ----  | ----  | ----  |
    | 含义  | 将激活量化成int8 | 不进行激活量化 |
    * `weight_quant_dtype`：mindspore.dtype类型，默认为mindspore.dtype.int8，可选输入及含义如下：

    |  weight_quant_dtype  | mindspore.dtype.int8（默认）  | None  |
    |  ----  | ----  | ----  |
    | 含义  | 将权重量化成int8 | 不进行权重量化 |
    * `kvcache_quant_dtype`：mindspore.dtype类型，默认为None，可选输入及含义如下：

    |  kvcache_quant_dtype  | mindspore.dtype.int8  | None（默认）  |
    |  ----  | ----  | ----  |
    | 含义  | 将kvcache量化成int8 | 不进行kvcache量化 |
* 新增 `OmniQuantConfig` 类，用于配置OmniQuant的algo_args。
* `NetworkHelper` 类新增 `get_decoder_layer` 、 `get_linears` 方法用于获取网络的decoder层及sub-cell的linear层；新增 `offload_embedding` 接口释放tensor占用的显存。
* `MFLlama2Helper`新增上述三个接口的实现，适配MindFormers中的llama2系列模型。

### 贡献者

感谢以下人员做出的贡献:

ccsszz, yyyyrf, hangangqiang

欢迎以任何形式对项目提供贡献！
