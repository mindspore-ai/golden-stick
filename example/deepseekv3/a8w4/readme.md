# DeepSeekR1网络A8W4量化算法指南

A8W4量化校准为层间组合量化，通过不同层的量化策略配置，使得DeepSeekR1模型可在单机8卡上部署，且精度劣化在1%以内。

逐层配置策略如下：

- MLA: 静态A8W8量化

- MLP: 动态A8W8量化

- MOE-路由专家：**动态A8W4量化**

- MOE-共享专家: 动态A8W8量化

## 进行算法校准阶段

本指南基于单机8卡，910服务器，每张卡可用显存60G左右。

量化校准使用当前目录下calibrate_deepseek3_671b.yaml，修改yaml文件的load_checkpoint、pretrained_model_dir为DeepSeekR1浮点模型的权重所在目录。

可通过如下命令一键拉起进程，进行A8W4算法校准：

```bash
bash calibrate.sh /path/to/mindformers 8 /path/to/save/model
```

校准结束后，通过如下脚本对权重进行合并：

```python
python unify_safetensors.py --input_dir=/path/to/save/model --output_dir=/path/to/save/model_unify --output_file_prefix=a8w4 --rank_num=8 --quant_type=a8w4
```

合并完成后，请将/path/to/save/model下的quantization_description.json文件复制到/path/to/save/model_unify目录下。
