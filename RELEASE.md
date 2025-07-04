# MindSpore Golden Stick Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Golden Stick 1.2.0 Release Notes

### Major Features and Improvements

* The Post-Training Quantization algorithm `PTQ` supports the MOE structure with `SmoothQuant-A8W8` quantization algorithm and `GPTQ-A16W4` low-bit quantization algorithm. These have been adapted for the DeepSeekV3/R1 model.

* Added OutlierSuppression-Lite(OSL), an outlier suppression technique. OSL is an extension of SmoothQuant that tunes the migration strength alpha through hyperparameter optimization at the matrix granularity to achieve more fine-grained network self-adaptive calibration.

* [Demo] The Post-Training Quantization algorithm `PTQ` supports the `A8W4` quantization algorithm. These have been adapted for the DeepSeekV3/R1 model.

* Added the loading and evaluation of the datasets `wikitext`, `boolq`, `ceval`, and `gsm8k`.

* Accuracy of DeepSeekR1:

    |  method  | ceval  | gsm8k  |
    |  ----  | ----  | ----  |
    | BF16  | 89.67 | 91.74 |
    | SmoothQuant-A8W8  | 89.45 | 92.42 |
    | OSL-A8W8  | 89.9 | 91.81 |
    | GPTQ-A16W4  | 89.52 | 91.12 |
    | A8W4 | 89.0 | 91.51 |

### Contributors

Thanks goes to these wonderful people:

tongl, zhuxiaochen, guoguopot, huangzhuo, ccsszz, yyyyrf, hangangqiang, HeadSnake

Contributions of any kind are welcome!

## MindSpore Golden Stick 1.0.0 Release Notes

### Major Features and Improvements

* The Post-Training Quantization algorithm `PTQ` supports the `GPTQ` quantization algorithm, which can quantize weight to 8 bit or 4 bit. `GPTQ` has been added to the precision recovery algorithm set and can be selected through `precision_recovery` in `PTQConfig`. Currently, only the `GPTQ` algorithm is available for precision recovery.
* The Post-Training Quantization algorithm `PTQ` supports the `AWQ` quantization algorithm. It enables `AWQ` by adding a new outlier suppression method to quantize the weights to 4 bits. The outlier suppression method can be selected through `outliers_suppression` in `PTQConfig`, and currently, two methods are available: `smooth` and `awq`.
* The Post-Training Quantization algorithm `PTQ` supports per-token dynamic quantization for activations, enabling online quantization of activations. This can be configured through `PTQConfig` with `act_quant_granularity=QuantGranularity.PER_TOKEN`.

### API Change

* The quantization methods `RoundToNearest` and `SmoothQuant` are deprecated. Please replace them with `PTQ`.

### Contributors

Thanks goes to these wonderful people:

huangzhuo, zhangminli, ccsszz, yyyyrf, hangangqiang

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.6.0 Release Notes

### Major Features and Improvements

* The `RoundToNearest` supports Mindformers' KVCache int8 quantization now, i.e. `PagedAttentionMgr` class, mainly for Llama2 networks.
* Added Post-Training Quantization algorithm named `PTQ` which supports SmoothQuant, A16W8, KVCacheInt8 and their combinations, such as A16W8 combined with KVCacheInt8, SmoothQuant combined with KVCacheInt8, etc., and the corresponding algorithm capabilities can be obtained by configuring PTQConfig. The algorithm is mainly supports ParallelLlama2 network from the MindFormers community.

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
    * `outliers_suppression`: The data type is OutliersSuppressionType. The default value is OutliersSuppressionType.NONE. The options and meanings are as follows:

        |  outliers_suppression  | OutliersSuppressionType.SMOOTH  | OutliersSuppressionType.NONE（default）  |
        |  ----  | ----  | ----  |
        | meanings  | employ smooth approach to suppress outliers in activation and weight | does not suppress outliers |

### Contributors

Thanks goes to these wonderful people:

ccsszz, yyyyrf, hangangqiang

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.5.0 Release Notes

### Major Features and Improvements

* [DEMO] Added post-training quantization W8A8 algorithm `SmoothQuant` mainly for Llama2 network.

### API Change

* Added `kwargs` to `apply` api of `CompAlgo` class as extensible parameter for subclasses.
* Added `NetworkHelper` abstract class as adapter for decoupling between algorithm and framework.
* Added `MFLlama2Helper` class as adapter between algorithm and MindFormers.
* [DEMO] Added `SmoothQuant` class as entry of SmoothQuant algorithm.
* Added parameter checking that `RoundToNearest` algorithm only supports BackendTarget.ASCEND as backend.

### Contributors

Thanks goes to these wonderful people:

ccsszz, yyyyrf, hangangqiang

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.4.1 Release Notes

### Major Features and Improvements

* Optimize the time taken by RoundToNearest algorithm to quantify weights.
* Optimize the compilation time of RoundToNearest quantization network.

### Contributors

Thanks goes to these wonderful people:

ccsszz, yyyyrf, hangangqiang

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.4.0 Release Notes

### Major Features and Improvements

* Added post-training weight quantization W8A16 algorithm `RoundToNearest`, which realizes the lossless compression parameters of Llama2 7B/13B/70B and Baichuan2 13B networks by over 40%.

### API Change

* Added `PTQConfig` to configure the post-training quantization algorithm.
* Added `PTQMode` enumeration class, which can be configured in 'PTQConfig', is used to distinguish between the two phases of the quantization algorithm: the quantization phase and the deployment phase.
* Added `BackendTarget` enumeration class, which can be configured in `PTQConfig`, to indicate the backend to which the quantized network will eventually be deployed. For example, 'BackendTarget.Ascend' indicates that it will eventually be deployed to the Ascend backend of MindSpore.

### Contributors

Thanks goes to these wonderful people:

zhuxiaoxiong, hangangqiang

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.3.0 Release Notes

### Bug fixes

* Fixed the problem that SCOP algorithm training fails to converge.

### Contributors

Thanks goes to these wonderful people:

hangangqiang, yangruoqi713, kevinkunkun.

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.3.0-alpha Release Notes

### Major Features and Improvements

* [stable] SLB（Searching for Low-Bit Weights in Quantized Neural Networks）QAT algorithm now support BatchNorm calibration. we can invoke `set_enable_bn_calibration` api to enable BatchNorm calibration. For a network with a BatchNorm layer, the BatchNorm calibration can reduces the decrease in network accuracy caused by the SLB quantization algorithm. ([!150](https://gitee.com/mindspore/golden-stick/pulls/150))
* [stable] We verified the quantization effect of SimQAT(Simulated Quantization Aware Training) algorithm and the SLB algorithm on the ResNet network and the Imagenet2012 dataset. For details, please refer to [MindSpore Models readme](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/official/cv/ResNet#%E7%BB%93%E6%9E%9C-4).
* [stable] SimQAT algorithm now support inference on MindSpore Lite backend. We quant the LeNet network with SimQAT and deploy it on ARM CPU. For details, please refer to [Deployment Effect](https://www.mindspore.cn/golden_stick/docs/en/master/quantization/simulated_quantization.html#summary-of-deployment).

### API Change

#### Backwards Compatible Change

* SLB algorithm adds the `set_enable_bn_calibration` interface to enable or disable BatchNorm calibration.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* Add `convert` interface to the algorithm base class, which is configured to convert training network to inferring network. And the network will be exported to MindIR file for Deployment. For details, please refer to [Model Deployment](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/deployment/convert.html#export-mindir-after-training).([!176](https://gitee.com/mindspore/golden-stick/pulls/176/files))
* Add `set_save_mindir` interface to the algorithm base class, which is configured to automatically export MindIR after training. For details, please refer to [Model Deployment](https://www.mindspore.cn/golden_stick/docs/en/r0.3.0-alpha/deployment/convert.html#configure-the-algorithm-to-automatically-export-mindir).([!168](https://gitee.com/mindspore/golden-stick/pulls/168/files))

### Bug fixes

* [STABLE] Refactor SimQAT algorithm code, and solve bugs such as activation operator loss, pre-trained parameter loss, simulation quantization operators redundancy, etc.

### Contributors

Thanks goes to these wonderful people:

liuzhicheng01, fuzhongqian, hangangqiang, yangruoqi713, kevinkunkun.

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.2.0 Release Notes

### Major Features and Improvements

* [STABLE] SLB(Searching for Low-Bit Weights in Quantized Neural Networks) QAT algorithm implements a built-in temperature adjustment callback to simplify the use of the algorithm. Users no longer need to manually write the temperature adjustment logic int the training script, and the original temperature adjustment function can be realized through the algorithm configuration interface. Note that this is an incompatible change.

### Bug fixes

* [STABLE] Solve a bug of AllReduce during distributed training, so that the SLB QAT algorithm can support distributed training.

### API Change

#### Backwards Incompatible Change

#### Python API

* Added `callbacks` interface to the algorithm base class, which returns the callback logic of the algorithm which will be called during the training process. In order to facilitate different algorithms to implement their own callback logic, this method has variable parameter inputs.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_epoch_size` interface, which is used to configure the total number of epochs of training, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_has_trained_epoch` interface. If a pre-trained checkpoint is used in training, it is used to configure the number of pre-trained epochs corresponding to the pre-trained checkpoint used in the current training, which is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_t_start_val` interface, which is used to configure the initialization value of the temperature in the temperature adjustment mechanism, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_t_start_time` interface, which is used to configure the time when the temperature adjustment mechanism start to work, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_t_end_time` interface, which is used to configure the time when the temperature adjustment mechanism stop to work, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))
* SLB algorithm adds the `set_t_factor` interface, which is used to configure the temperature adjustment factor in the temperature adjustment mechanism, and is used to implement the temperature adjustment callback logic.([!117](https://gitee.com/mindspore/golden-stick/pulls/117))

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368, yangruoqi713, kevinkunkun.

Contributions of any kind are welcome!

## MindSpore Golden Stick 0.1.0 Release Notes

MindSpore Golden Stick is a model compression algorithm set jointly designed and developed by Huawei's Noah team and Huawei's MindSpore team. MindSpore Golden Stick provides an unified user interface allowing users to apply model compression algorithms such as quantization and pruning in a unified and convenient manner. MindSpore Golden Stick also provides front-end network modification capabilities to reduce algorithm development costs. MindSpore Golden Stick provides three algorithms in current version.

### Major Features and Improvements

* [BETA] Provides a quantization aware training algorithm named SimQAT (Simulated Quantization Aware Training), which is the most basic quantization aware training algorithm.
* [BETA] Provides a quantization aware training algorithm called SLB (Searching for Low-Bit Weights in Quantized Neural Networks), which is a nonlinear, high-precision quantization aware training algorithm with obvious advantages in low-bit quantization.
* [STABLE] Provides a pruning algorithm named SCOP (Scientific Control for Reliable Neural Network Pruning), which is a high-precision structured pruning algorithm and is mainly used in CV networks at present.

### Contributors

Thanks goes to these wonderful people:

ghostnet, liuzhicheng01, fuzhongqian, hangangqiang, cjh9368.

Contributions of any kind are welcome!
