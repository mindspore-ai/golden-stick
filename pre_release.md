
## MindSpore Golden Stick 0.6.0 Release Notes

### Major Features and Improvements

* The `RoundToNearest` supports Mindformers' KVCache int8 quantization now, i.e. `PagedAttentionMgr` class, mainly for Llama2 networks.
* Added `SmoothQuant` A8W8 PTQ algorithm aimed to quant Linear of MindFormers, mainly for Llama2 networks.
* Added pynative-based algorithm `OmniQuant`, which supports AutoClip and AutoScale, mainly for llama2 series networks. The user can set the hyperparameters related to Clip and Scale to the list or float type to determine whether to perform parameter search or network quantization.
* Added `load_mindformers_plugin` static method in `RoundToNearest` for decoupling `RoundToNearest` with MindFormers. If user want to quantize network from MindFormers, please invoke this method before `RoundToNearest` being created.

### API Change

* `PTQConfig` adds the following three parameters:
    * `act_quant_dtype`: The data type is mindspore.dtype. The default value is None. The options and meanings are as follows:

    |  act_quant_dtype  | mindspore.dtype.int8  | None(default)  |
    |  ----  | ----  | ----  |
    | meanings  | quantize input to int8 | does not quantize input |
    * `weight_quant_dtype`: The data type is mindspore.dtype. The default value is mindspore.dtype.int8. The options and meanings are as follows:

    |  weight_quant_dtype  | mindspore.dtype.int8(default)  | None  |
    |  ----  | ----  | ----  |
    | meanings  | quantize weights to int8 | does not quantize weights |
    * `kvcache_quant_dtype`: The data type is mindspore.dtype. The default value is None. The options and meanings are as follows:

    |  kvcache_quant_dtype  | mindspore.dtype.int8  | None（default）  |
    |  ----  | ----  | ----  |
    | meanings  | quantize kvcache to int8 | does not quantize kvcache |
* Added `OmniQuantConfig` class for configuring algo_args of OmniQuant algorithm.
* Added `get_decoder_layer`, `get_linears` method to `NetworkHelper` class to obtain the decoder layer of network and the linear layer of sub-cell.
* Added implementation of `get_decoder_layer` , `get_linears` and `offload_embedding` of `MFLlama2Helper` to work with the Llama2 series models in MindFormers.
* Added `MFParallelLlama2Helper` class to work with the ParallelLlamaForCasualLM model in Mindformers. Added implementation of `create_network` to create ParallelLlamaForCasualLM model. Added implementation of `get_decoder_layer` and `get_linears` to obtain the decoder layer of network and linear layer of decoder.

### Contributors

Thanks goes to these wonderful people:

ccsszz, yyyyrf, hangangqiang

Contributions of any kind are welcome!
