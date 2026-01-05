# 安装MindSpore Golden Stick

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/golden-stick/blob/master/docs/zh_cn/install.md)

[View English](../en/install.md)

## 环境限制

下面列出了安装、编译和运行MindSpore Golden Stick所需的系统环境：

- 操作系统：openEuler/Ubuntu/Linux
- Python：>=3.9, <3.12
- 硬件：建议使用昇腾 NPU（Atlas 800I A2或更高，不支持Atlas 300系列）

> 其他的三方依赖请参考[requirements文件](https://gitee.com/mindspore/golden-stick/blob/master/requirements.txt)。

## 版本生命周期及版本配套策略

MindSpore Golden Stick版本有以下五个维护阶段：

|   **状态**    | **期限** | **说明**                         |
|:-----------:|:------:|:-------------------------------|
|     计划      | 1-3 个月 | 规划功能。                          |
|     开发      |  3 个月  | 构建功能。                          |
|     维护      |  6 个月  | 合入所有已解决的问题并发布新版本。              |
|     无维护     | 0-3 个月 | 合入所有已解决的问题，没有专职维护团队，且不计划发布新版本。 |
| 生命周期终止（EOL） |  N/A   | 分支进行封闭，不再接受任何修改。               |

### 历史版本生命周期

| MindSpore Golden Stick版本 | 当前状态  |   发布时间  |         后续状态         |   EOL日期   |
| :---------------------: | :--------: | :--------: | :--------------------: | :--------: |
|          1.4.1          |  维护       | 2026-01-08 | 预计2026-07-08终止维护   | 2026-10-08 |
|          1.4.0          |  维护       | 2026-01-04 | 预计2026-07-04终止维护   | 2026-10-04 |
|          1.3.0          |  维护       | 2025-10-23 | 预计2026-04-23终止维护   | 2026-07-23 |
|          1.2.0          |  维护       | 2025-08-13 | 预计2026-02-13终止维护   | 2026-05-13 |
|          1.1.0          | 无维护      | 2025-05-21 | 预计2026-02-21终止生命周期| 2026-02-21 |
|          1.0.0          | 生命周期终止 | 2025-02-13 |                        | 2025-11-13 |
|          0.6.0          | 生命周期终止 | 2024-10-30 |                        | 2025-07-30 |
|          0.5.0          | 生命周期终止 | 2024-08-01 |                        | 2025-05-01 |
|          0.4.1          | 生命周期终止 | 2024-07-15 |                        | 2025-04-15 |
|          0.4.0          | 生命周期终止 | 2024-03-30 |                        | 2024-12-30 |
|          0.3.0          | 生命周期终止 | 2023-06-15 |                        | 2024-03-15 |
|       0.3.0-alpha       | 生命周期终止 | 2023-02-01 |                        | 2023-11-01 |
|          0.2.0          | 生命周期终止 | 2022-10-26 |                        | 2023-07-26 |
|          0.1.0          | 生命周期终止 | 2022-07-29 |                        | 2023-04-29 |

## 版本依赖映射表

MindSpore Golden Stick 对 MindSpore、MindOne 以及 MindFormers 存在版本依赖关系，具体如下表所示：

| MindSpore Golden Stick版本 |                             分支                                  | MindSpore版本 | MindSpore Transformers版本 |
| :---------------------: | :-----------------------------------------------------------------: | :----------: | :------------------------: |
|          1.4.1          | [r1.4](https://gitee.com/mindspore/golden-stick/tree/r1.4.0/)       |  2.7.1.post1 |       1.7.0                |
|          1.4.0          | [r1.4](https://gitee.com/mindspore/golden-stick/tree/r1.4.0/)       |   2.7.1      |       1.7.0                |
|          1.3.0          | [r1.3](https://gitee.com/mindspore/golden-stick/tree/r1.3.0/)       |   2.7.1      |       1.7.0                |
|          1.2.0          | [r1.2](https://gitee.com/mindspore/golden-stick/tree/r1.2.0/)       |   2.7.0      |       1.6.0                |
|          1.1.0          | [r1.1](https://gitee.com/mindspore/golden-stick/tree/r1.1.0/)       |   2.6.0      |       1.5.0                |
|          1.0.0          | [r1.0](https://gitee.com/mindspore/golden-stick/tree/r1.0.0/)       |   2.5.0      |       1.4.0-beta2          |
|          0.6.0          | [r0.6](https://gitee.com/mindspore/golden-stick/tree/r0.6.0/)       |   2.4.0      |       1.3.0                |
|          0.5.0          | [r0.5](https://gitee.com/mindspore/golden-stick/tree/r0.5.0/)       |   2.3.1      |       1.2.0                |
|          0.4.1          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4.1/)       |   2.3.0      |       1.2.0                |
|          0.4.0          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4/)         |   2.3.0-rc1  |        NA                  |
|          0.3.0          | [r0.3](https://gitee.com/mindspore/golden-stick/tree/r0.3/)         |   2.0.0-rc1, 2.0.0   |    NA              |
|       0.3.0-alpha       | [r0.3](https://gitee.com/mindspore/golden-stick/tree/v0.3.0-alpha/) |   2.0.0-alpha        |        NA          |
|          0.2.0          | [r0.2](https://gitee.com/mindspore/golden-stick/tree/r0.2/)         |   1.9.0      |        NA                  |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/)         |   1.8.0      |        NA                  |

> 金箍棒早期版本不涉及MindSpore Transformers，故表中标记为NA。

## 安装CANN

1. 根据版本依赖映射表，查询对应的 MindSpore 版本；
2. 根据 [MindSpore 官网](https://www.mindspore.cn/versions) 查询推荐的 CANN 版本；
3. 从 [CANN 开发者网站](https://www.hiascend.com/hardware/firmware-drivers/community) 下载安装驱动和固件；
4. 根据 [CANN 安装指引](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit) 安装并配置 CANN 环境。

## 安装MindSpore

- **1.4.0 版本及以后**：MindSpore 已加入到 MindSpore Golden Stick 的 `requirements.txt` 中，用户不再需要手动安装该依赖。
- **1.4.0 版本以前**：用户需要根据版本依赖映射表，使用 pip 安装或从 [MindSpore 官网](https://www.mindspore.cn/install) 下载对应的安装包进行安装。

## 安装MindFormers

- **1.4.0 版本及以后**：MindFormers（原 MindSpore Transformers）不再是 MindSpore Golden Stick 的必需依赖项。仅当需要量化来自 MindFormers 生态的模型时，用户可根据版本依赖映射表安装对应的 MindFormers 版本。
- **1.4.0 版本以前**：用户需要根据版本依赖映射表，使用 pip 安装对应的 MindFormers 版本。

```bash
pip install mindformers==1.7.0
```

## 安装MindOne

**1.4.0 版本及以后**：MindSpore Golden Stick 支持对来自 MindOne 生态的模型进行量化，作为依赖项引入到MindSpore Golden Stick中。由于 MindOne 中不同模型可能在不同版本中支持，如果相应的模型有[Model Card](https://gitee.com/mindspore/golden-stick/tree/master/example/model_cards) ，请优先使用其中 README 安装对应的 MindOne 版本。

## 安装MindSpore Golden Stick

可以通过pip安装或者源码编译方式安装MindSpore Golden Stick。

### pip安装

我们在pypi上维护了[MindSpore Golden Stick项目](https://pypi.org/project/mindspore-gs/)，可以直接使用pip命令安装。

```shell
pip install mindspore-gs
```

### 源码编译安装

下载[源码](https://gitee.com/mindspore/golden-stick)，下载后进入`golden_stick`目录。

```shell
git clone https://gitee.com/mindspore/golden-stick.git
cd golden-stick
bash build.sh
```

编译成功后会在output目录下生成whl包，使用pip安装即可。

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_gs
import mindspore_gs.ptq
```
