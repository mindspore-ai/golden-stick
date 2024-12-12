# 应用PTQ算法

[View English](./README.md)

## PTQ算法介绍

### 设计初衷

为了实现组合量化算法，以及后续金箍棒对于混合精度量化、自动搜优等复杂量化算法的支持，金箍棒引入名为PTQ的训练后量化算法。该算法依赖于MindSpore前端并行能力，得以实现更加复杂的算法实现逻辑。

该算法能够提供RoundToNearest和SmoothQuant两个量化算法的能力，后续新的训练后量化算法也会在此算法上演进，所以我们将该算法命名为PTQ。

### 设计思路

![架构图](images/zh_cn/arch.png)

分层实现量化算法，主要分为config、量化算法、算法模块、量化Cell、量化工具函数。

- config主要用于用户配置算法。并且实现yaml序列化反序列化能力。
- 量化算法是算法的主入口，PTQ算法同样继承自金箍棒算法基类CompAlgo，实现了apply和convert接口，分别实现量化checkpoint和量化部署网络导出的功能。
- 算法模块是一些模块化的功能块，比如本次PTQ算法中内置了针对Linear层的Smooth模块，针对Linear和KVCache层的量化模块。通过这些模块的组装，可以实现各种算法，比如SmoothQuant算法。这保证了PTQ算法高度的扩展性和灵活性。
- 量化Cell是针对特定的非量化网络层，封装得到的量化网络层，用于实现对特定网络层的量化。量化网络层通过注册方式引入，实现了不同网络框架之间的解耦，比如金箍棒和MindFormers解耦。
- 量化工具函数是一些基础的工具函数，如量化参数的计算，矩阵的量化等。

### 使用限制

表1：PTQ算法规格

| 规格 | 规格说明 |
| --- | --- |
| 硬件支持 | Atlas 800I A2 |
| 网络支持 | ParallelLlamaForCausalLM，具体请参见[ParallelLlamaForCausalLM网络](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/experimental/infer/models/llama/llama.py) |
| 运行模式支持 | 量化checkpoint阶段仅支持PyNative模式，量化推理阶段不限定模式，建议GraphMode获得更好的性能 |

> 当前PTQ算法依赖于完整的DecoderLayer做网络拓扑分析，所以不支持任意基于MindFormers的Linear层构造的网络，我们计划在后续版本改进这一点，以提升PTQ算法的网络泛化能力。

### 算法支持

训练后量化算法有很多种分类维度，比如静态量化和动态量化；权重量化、激活量化和KVCache量化；MinMax量化、MSE量化、KL散度量化和直方图量化；还有各种量化的优化技术，从最简单的四舍五入量化，到SmoothQuant量化，GPTQ量化，AWQ量化等。

本小节从业界常见的量化算法范式来介绍PTQ算法的能力，在此之前先给出其他分类维度上的一些限制：

- 仅支持MinMax量化。
- 仅支持静态量化，其中激活量化只支持per-tensor量化，权重量化只支持per-channel量化。
- 受限于硬件和算子支持，对于全量化，激活不支持per-channel的量化，权重不支持带zero point的量化。
- 硬件支持带zero point的权重量化，但当前PTQ算法没有开放这方面能力，仅支持不带zero point的权重量化。
- PTQ算法做了分层设计，当前底层量化算子仅对MindFormers的一些Layer做了支持，因为PTQ算法仅支持对[MindFormers的Linear层](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/modules/layers.py#L363)做权重量化和激活量化，对[MindFormers的PageAttention层](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/modules/paged_attention_mgr.py#L26)做KVCache量化。如果用户需要量化不基于MindFormers的网络，需要用户提供相关量化算子实现，当前这方面自定义能力没有形成明确的接口，会在未来提供。

#### RoundToNearest算法

RoundToNearest算法是一类较朴素的后量化算法，其取整方式使用了Round to nearest，即四舍五入的方式，故名RoundToNearest。该算法能力和金箍棒独立的[RoundToNearest](../round_to_nearest/README_CN.ipynb)算法能力类似，后续金箍棒会停止对RoundToNearest算法的演进，使用PTQ算法来支持RoundToNearest算法能力。

##### 1）量化过程

![](images/zh_cn/round_to_nearest.png)

量化算法的主要逻辑是根据浮点数据如权重的最大最小值和整型数据的最大最小值，根据计算公式计算量化参数：

$$scale = \frac{X_{{float}_{max}} - {X_{float}}_{min}} {X_{{int}_{max}} - {X_{int}}_{min}}$$

$$offset = round(X_{{int}_{max}} - \frac{X_{{float}_{max}}} {scale})$$

其中scale是缩放因子，offset是平移因子，两者统称为量化参数。获得量化参数后就可以对权重做量化：

$$x_{int} = clamp(round(x_{float} \div scale) + offset; 0, 2^b-1)$$

其中涉及round操作，即四舍五入操作，这是RoundToNearest算法的含义，也是该量化算法的一部分误差来源。

##### 2）权重量化

将上述量化过程应用于网络中的权重矩阵，将其转换为8bit整型进行存储。在部署时加载8bit权重后，对其进行反量化，其过程的数学表达如下：

$$X_{float} = (X_{int} - offset) \times scale$$

将权重反量化为浮点后，网络的推理过程就和一般的浮点网络推理过程无异。权重量化并不能带来计算量的减少，相反反量化会带来额外的计算量，所以通常将反量化的操作和后续的浮点计算过程进行融合，可以有效降低部署阶段的显存开销，同时可以缓解大语言模型增量推理阶段的Memory Bound，这两者都可以提升大语言模型部署时的吞吐量。

PTQ当前仅支持8bit的权重量化能力，可以通过如下配置项使能：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8,  act_quant_dtype=None,  kvcache_quant_dtype=None,
                        outliers_suppression=OutliersSuppressionType.NONE)
```

##### 3）KVCache量化

将上述量化过程应用于大语言模型推理过程中产生的KVCache上，将计算得到的KVCache量化后进行存储，然后在Attention计算前将KVCache进行反量化，从而缓解KVCache的显存占用，以支持更大batch size或者更长序列的大语言模型生成。需要注意的是，KVCache是推理时产生的，所以针对KVCache的量化需要少量数据集进行校准。

PTQ当前仅支持8bit的KVCache量化能力，可以通过如下配置项使能：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=None, act_quant_dtype=None, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.NONE)
```

#### SmoothQuant算法

研究发现，不同于CNN和小型的transformer网络，当大语言模型参数量超过6.8B时，网络的激活中出现“systematic outliers with large magnitude”，由于浮点的分布很广且不均匀，导致难以量化。

![](images/zh_cn/smooth_quant.png)

SmoothQuant算法通过数学等价变换，将激活上的异常值转移一部分到权重上，从而将难以量化的激活和极易量化的权重转化为较易量化的激活和较易量化的权重，实现量化精度的提升。

可以通过如下配置项使能PTQ的SmoothQuant能力：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
```

##### GPTQ算法

[GPTQ](https://arxiv.org/abs/2210.17323)算法是一种针对大规模预训练模型的PTQ算法。其核心思想是在量化过程中对权重weight进行补偿，从而减少低bit量化导致模型精度的损失。

PTQ算法支持使用GPTQ算法进行4bit权重量化，并将其添加到了精度恢复算法集中，精度恢复算法当前仅GPTQ算法可选。

可以通过如下配置项使能PTQ的GPTQ算法:

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType, PrecisionRecovery
from mindspore_gs.ptq.ptq_config import GPTQQuantConfig

algorithm_config = GPTQQuantConfig()
ptq_config = PTQConfig(weight_quant_dtype=qint4x2, act_quant_dtype=None, kvcache_quant_dtype=None,
                       outliers_suppression=OutliersSuppressionType.NONE, algo_args=algorithm_config,
                       precision_recovery=PrecisionRecovery.GPTQ)
```

#### 组合量化

得益于分层解耦框架设计，PTQ算法可以方便的将不同的算法能力组合在一起：

- 8bit权重量化组合8bit KVCache量化：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=None, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.NONE)
```

- SmoothQuant量化组合8bit KVCache量化：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.NONE)
```

后续我们还将基于此支持层间混合精度量化，根据不同层对于量化的敏感程度，应用8bit权重量化、8bit全量化等。

## 示例

跟金箍棒所有算法一样，PTQ算法的应用主要可以分为两个阶段：量化阶段和部署阶段。

量化阶段是部署前提前完成的，主要的工作是：收集权重的分布、计算量化参数、量化权重数据、插入反量化节点。

部署阶段通常是指在生产环境，使用MindSpore框架对量化后的模型进行推理的过程。

本用例使用ParallelLlamaForCausalLM 7B网络进行演示，主要分四个步骤：环境准备、模型量化、模型部署评估、效果分析。

### 步骤1. 环境准备

#### 1.1. Ascend环境

PTQ算法需要运行在Ascend硬件上，Ascend的环境配置可以参考[MindSpore安装指南](https://www.mindspore.cn/install)安装昇腾AI处理器配套软件包小节和配置环境变量小节。

#### 1.2. MindSpore环境

金箍棒依赖于MindSpore，需要提前安装合适的MindSpore。可以从MindSpore官网下载预编译好的[v2.4.0版本安装包](https://www.mindspore.cn/versions)并安装。

#### 1.3. MindFormers环境

本样例对MindFormers中的网络进行量化并推理，所以需要提前安装合适的MindFormers。可以从MindSpore官网下载预编译好的[v1.3.0版本安装包](https://www.mindspore.cn/versions)并安装。

#### 1.4. 金箍棒环境

从MindSpore官网下载预编译好的[MindSpore GoldenStick v0.6.0 版本安装包](https://www.mindspore.cn/versions)并安装。

#### 1.5. 相关文件准备

需要预先下载[squad1.1数据集](https://data.deepai.org/squad1.1.zip)、[Llama2 7B预训练权重](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)和[Llama2分词器文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)。

**第一步**创建工作目录：

```shell
mkdir workspace
```

**第二步**准备数据集，由于权限限制，需要手动下载squad数据集：

数据集下载地址：[squad1.1数据集](https://data.deepai.org/squad1.1.zip)

下载完成后，将得到的数据集squad1.1.zip拷贝至第一步创建的workspace目录下，并确保数据集名称为squad1.1.zip，然后运行解压代码：

```shell
cd workspace
unzip squad1.1.zip -d ./squad
```

使用unzip命令解压squad1.1.zip文件后，可以得到train-v1.1.json和dev-v1.1.json量化数据集文件，我们先使用train数据集进行量化校准，然后使用dev数据集进行量化评测。

**第三步**准备Llama2 7b网络的checkpoint文件，Llama2分词器文件，Llama2模型配置文件：

下载地址：

[Llama2 7b checkpoint](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt)

[Llama2分词器文件](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

[llama2模型配置文件](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/predict_llama2_7b.yaml)

下载好上述3个文件后，将其拷贝至workspace目录下。

准备完上述文件后，目录结构为：

```shell
workspace
  ├── squad
  ├     ├── train-v1.1.json
  ├     └── dev-v1.1.json
  ├── predict_llama2_7b.yaml
  ├── tokenizer.model
  └── llama2_7b.ckpt
```

### 步骤2. 模型量化

#### 2.1. 构造非量化网络

构造MindFormers仓的ParallelLlamaForCausalLM 7B网络，首先需要修改predict_llama2_7b.yaml文件的如下内容：

1. 更新load_checkpoint字段为llama2_7b.ckpt所在路径。

2. 更新process中的vocab_file字段为tokenizer.model所在路径。没有该字段的话，可手动添加。

3. 修改context中的device_id为当前机器空闲的设备id，context中的mode为1，即PYNATIVE模式。

4. 修改model.arch.type字段为ParallelLlamaForCausalLM。

5. 修改use_parallel为True, parallel.parallel_mode为'STAND_ALONE'，parallel_config.data_parallel为1，parallel.full_batch为False。

修改后的yaml配置文件中，并行相关的配置应该类似于这样：

```yaml
use_parallel: True
parallel:
  parallel_mode: "STAND_ALONE"
  gradients_mean: False
  enable_alltoall: False
  full_batch: False
  search_mode: "sharding_propagation"
  enable_parallel_optimizer: False
  strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
  parallel_optimizer_config:
    gradient_accumulation_shard: False
    parallel_optimizer_threshold: 64
parallel_config:
  data_parallel: 1
  model_parallel: 1
  pipeline_stage: 1
  use_seq_parallel: False
  micro_batch_num: 16
  vocab_emb_dp: True
  gradient_aggregation_group: 4
```

需要注意的是MindFormers的ParallelLlamaForCausalLM 7B网络即便在单卡上，也必须使用msrun才能成功运行。msrun的使用方式可以参考[msrun使用说明](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html)，下面是一个简单的样例命令：

```bash
msrun --worker_num=1 --local_worker_num=1 --master_port=12345 --log_dir=msrun_log --join=True --cluster_time_out=300 python sample.py
```

完整的样例代码可以参考[quant_ckpt.py](https://gitee.com/mindspore/golden-stick/blob/master/example/ptq/quant_ckpt.py)。

修改完成后，可以使用金箍棒提供的MFParallelLlama2Helper方便地通过配置文件构造网络并加载checkpoint，代码如下：

```python
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper

config_path = '/path/to/workspace/predict_llama2_7b.yaml'
helper = MFParallelLlama2Helper(config_path)
network = helper.create_network()
```

#### 2.2. 构造squad-v1.1数据集loader

我们基于squad的train-v1.1.json进行量化过程中的校准，用mindspore_gs的get_datasets接口构造squad-v1.1数据集loader。

一般量化校准阶段只会使用数百条数据进行校准，当前样例中，我们使用n_samples参数指定仅加载数据集中的200条数据，代码如下：

```python
from mindspore_gs.datasets import get_datasets

ds_path = '/path/to/workspace/squad/train-v1.1.json'
bs_ = helper.get_spec('batch_size')
seq_ = helper.get_spec('seq_length')
max_decode_length = helper.get_spec('max_decode_length')
ignore_token_id = helper.get_spec('ignore_token_id')
tokenizer = helper.create_tokenizer()
ds = get_datasets('squad1.1', ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                  False, n_samples=200)
```

#### 2.3. 构造量化算法

PTQ算法支持基础的round to nearest方法实现的a16w8权重量化和c8（kvcache int8）算法，也支持smooth-quant方法实现的a8w8算法，同时也支持a16w8权重量化算法和c8算法组合量化算法，smooth-quant和c8组合量化算法。

我们可以根据PTQConfig配置来启用不同的量化能力，PTQConfig的含义可以参考其[API文档](https://www.mindspore.cn/golden_stick/docs/zh-CN/master/ptq/mindspore_gs.ptq.PTQConfig.html#mindspore_gs.ptq.PTQConfig)，这里我们以SmoothQuant为例：

```python
from mindspore import dtype as msdtype
from mindspore_gs.ptq import PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=None,
                    outliers_suppression=OutliersSuppressionType.SMOOTH)
```

有了PTQConfig以后，接下来构造PTQ算法了，代码如下：

> 对于ParallelLlamaForCausalLM网络，某些层对于量化比较敏感，不适合量化，我们通常通过opname_blacklist字段来帮助跳过这些层的量化。

```python
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType

ptq_config = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, opname_blacklist=["w2", "lm_head"],
                       weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8, kvcache_quant_dtype=msdtype.int8)
ptq = PTQ(config=ptq_config)
```

#### 2.4. 量化网络并保存量化checkpoint文件

接下来对网络进行量化矫正，主要分为两个步骤：**第一步**是使用PTQ的apply接口，对网络进行量化矫正；**第二步**是使用PTQ的convert接口，将量化矫正后的网络改造成对应后端的真实量化网络：

```python
import mindspore as ms

ptq.apply(network, helper, ds)
ptq.convert(network)
ms.save_checkpoint(network.parameters_dict(), "a8w8c8.ckpt",
                   choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
print("quant checkpoint saved at 'a8w8c8.ckpt'", flush=True)
```

成功运行后，量化后的checkpoint文件会保存在 `/path/to/workspace/a8w8c8.ckpt` 路径下。

### 步骤3. 模型部署

#### 3.1. 评估FP16网络的F1EM指标

使用squad1.1 dev数据集评估ParallelLlamaForCausalLM-7B网络的F1EM指标。完整样例可以参考[eval_squad.py](https://gitee.com/mindspore/golden-stick/blob/master/example/ptq/eval_squad.py)。注意需用msrun运行，msrun的使用方式可以参考[msrun使用说明](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html)。

> 评测前请确认yaml配置文件中的load_checkpoint字段已正确配置了非量化的网络checkpoint文件路径:`/path/to/workspace/llama2_7b.ckpt`。并配置context.mode为0，即静态图模式。

```python
import numpy as np
import mindspore as ms
from mindformers.core.metric import EmF1Metric
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
from mindspore_gs.datasets import get_datasets
from mindspore_gs.common import logger

config_path = '/path/to/workspace/predict_llama2_7b.yaml'
helper = MFParallelLlama2Helper(config_path)
network = helper.create_network()

ds_path = '/path/to/workspace/squad/dev-v1.1.json'
bs_ = helper.get_spec('batch_size')
seq_ = helper.get_spec('seq_length')
max_decode_length = helper.get_spec('max_decode_length')
ignore_token_id = helper.get_spec('ignore_token_id')
top_k = helper.get_spec("top_k")
top_p = helper.get_spec("top_p")
do_sample = helper.get_spec("do_sample")
pad_token_id = helper.get_spec("pad_token_id")
tokenizer = helper.create_tokenizer()
ds = get_datasets('squad1.1', ds_path, "eval", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                  False, n_samples=1000)

metric = EmF1Metric()
metric.clear()

data_count = 0
total_count = ds.get_dataset_size()
for _, ds_item in enumerate(ds.create_dict_iterator()):
    data_count += 1
    logger.info(f"Dataset count: {data_count}/{total_count}")
    input_ids = ds_item['input_ids'].asnumpy()
    labels = ds_item['labels'].asnumpy()

    valid_length_each_example = []
    for j in range(input_ids.shape[0]):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
    valid_length_each_example = np.array(valid_length_each_example)

    outputs = network.generate(input_ids, do_sample=do_sample, max_length=seq_, top_p=top_p, top_k=top_k, max_new_tokens=200)
    output_ids = []
    for j in range(input_ids.shape[0]):
        output_ids.append(outputs[j][int(valid_length_each_example[j]):])

    pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
    labels_str = tokenizer.decode(labels, skip_special_tokens=True)
    metric.update(pres_str, labels_str)
metric.eval()
```

#### 3.2. 评估量化后网络的F1EM指标

由于MindSpore当前不支持保存修改后的网络，所以在加载量化ckpt之前，需要先用算法恢复带量化结构的网络，然后再加载checkpoint到网络。

评估脚本逻辑与非量化网络的一致，不过中间增加一步修改网络为量化网络的过程。

> 评测前请确认yaml配置文件中的load_checkpoint字段已经配置了正确的量化的网络checkpoint文件路径: `/path/to/workspace/a8w8c8.ckpt`。

```python
import numpy as np
import mindspore as ms
from mindspore.communication.management import init
from mindformers.core.metric import EmF1Metric
from mindformers import MindFormerConfig, AutoModel
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
from mindspore_gs.datasets import get_datasets
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType


config_path = '/path/to/workspace/predict_llama2_7b.yaml'
mf_config = MindFormerConfig(config_path)

ms.set_context(mode=mf_config.context.mode, device_target=mf_config.context.device_target,
                jit_config={"jit_level": "O0", "infer_boost": "on"})
init()
network = AutoModel.from_config(mf_config, download_checkpoint=False)
network.set_train(False)
network.phase = 'predict'

ptq_config = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=["w2", "lm_head"],
                       weight_quant_dtype=ms.dtype.int8, act_quant_dtype=ms.dtype.int8, kvcache_quant_dtype=ms.dtype.int8)
ptq = PTQ(config=ptq_config)
ptq.apply(network)
ptq.convert(network)

ms.load_checkpoint(mf_config.load_checkpoint, network)

helper = MFParallelLlama2Helper(mf_config)
ds_path = '/path/to/squad/dev-v1.1.json'
bs_ = helper.get_spec('batch_size')
seq_ = helper.get_spec('seq_length')
max_decode_length = helper.get_spec('max_decode_length')
ignore_token_id = helper.get_spec('ignore_token_id')
top_k = helper.get_spec("top_k")
top_p = helper.get_spec("top_p")
do_sample = helper.get_spec("do_sample")
pad_token_id = helper.get_spec("pad_token_id")
tokenizer = helper.create_tokenizer()
ds = get_datasets('squad1.1', ds_path, "eval", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                  False, n_samples=1000)

metric = EmF1Metric()
metric.clear()

data_count = 0
total_count = ds.get_dataset_size()
for _, ds_item in enumerate(ds.create_dict_iterator()):
    data_count += 1
    logger.info(f"Dataset count: {data_count}/{total_count}")
    input_ids = ds_item['input_ids'].asnumpy()
    labels = ds_item['labels'].asnumpy()

    valid_length_each_example = []
    for j in range(input_ids.shape[0]):
        # As the nonzero returns the index and we need length
        valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
    valid_length_each_example = np.array(valid_length_each_example)

    outputs = network.generate(input_ids, do_sample=do_sample, max_length=seq_, top_p=top_p, top_k=top_k, max_new_tokens=200)
    output_ids = []
    for j in range(input_ids.shape[0]):
        output_ids.append(outputs[j][int(valid_length_each_example[j]):])

    pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
    labels_str = tokenizer.decode(labels, skip_special_tokens=True)
    metric.update(pres_str, labels_str)
metric.eval()
```

### 步骤4. 效果分析

表2：ParallelLlamaForCausalLM-7B网络使用PTQ算法进行A8W8C8量化前后对比

| 指标 | FP16 | PTQ-A8W8C8 | 收益 |
| --- | --- | --- | --- |
| ckpt-size(GB)↓ | 13 | 7.9 | 39.2% |
| F1↓ | 33% | 32% | -1% |
| EM↓ | 0 | 0 | - |

可以看到，经过PTQ算法进行A8W8C8量化后：

1. 量化后网络的参数量缩减了39.2%，即网络部署时，用于静态显存占用下降到Float16时的60.8%。因而量化后的网络可以在资源更紧张的环境上部署，或者在相同的环境中提供更大的吞吐量。
2. 量化后网络的在squad1.1数据集上的F1下降1%，即量化后网络在squad1.1数据集判别式任务上效果略有下降。
