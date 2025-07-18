# 安装MindSpore Golden Stick

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/golden-stick/blob/master/docs/zh_cn/install.md)

[View English](../docs_en/install.md)

## 环境限制

下表列出了安装、编译和运行MindSpore Golden Stick所需的系统环境：

| 软件名称 |  版本   |
| :-----: | :-----: |
| Ubuntu  |  18.04  |
| Python  |  3.9-3.10 |

> 其他的三方依赖请参考[requirements文件](https://gitee.com/mindspore/golden-stick/blob/master/requirements.txt)。
> 当前MindSpore Golden Stick仅能在Ubuntu18.04上运行。

## MindSpore版本依赖关系

MindSpore Golden Stick依赖MindSpore训练推理框架，请按照根据下表中所指示的对应关系，并参考[MindSpore安装指导](https://mindspore.cn/install)安装对应版本的MindSpore：

| MindSpore Golden Stick版本 |                             分支                                  | MindSpore版本 |
| :---------------------: | :-----------------------------------------------------------------: | :-------: |
|          1.1.0          | [r1.1](https://gitee.com/mindspore/golden-stick/tree/r1.1.0/)       |   2.6.0   |
|          1.0.0          | [r1.0](https://gitee.com/mindspore/golden-stick/tree/r1.0.0/)       |   2.5.0   |
|          0.6.0          | [r0.6](https://gitee.com/mindspore/golden-stick/tree/r0.6.0/)       |   2.4.0   |
|          0.5.0          | [r0.5](https://gitee.com/mindspore/golden-stick/tree/r0.5.0/)       |   2.3.1   |
|          0.4.1          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4.1/)       |   2.3.0   |
|          0.4.0          | [r0.4](https://gitee.com/mindspore/golden-stick/tree/r0.4/)         |   2.3.0-rc1   |
|          0.3.0          | [r0.3](https://gitee.com/mindspore/golden-stick/tree/r0.3/)         |   2.0.0-rc1, 2.0.0   |
|       0.3.0-alpha       | [r0.3](https://gitee.com/mindspore/golden-stick/tree/v0.3.0-alpha/) |   2.0.0-alpha   |
|          0.2.0          | [r0.2](https://gitee.com/mindspore/golden-stick/tree/r0.2/)         |   1.9.0   |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/golden-stick/tree/r0.1/)         |   1.8.0   |

安装完MindSpore后，继续安装MindSpore Golden Stick。可以采用pip安装或者源码编译安装两种方式。

## pip安装

使用pip命令安装，请从[MindSpore Golden Stick下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

 ```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{ms_version}/GoldenStick/any/mindspore_gs-{mg_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Golden Stick安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{ms_version}`表示与MindSpore Golden Stick匹配的MindSpore版本号，例如下载1.0.0版本MindSpore Golden Stick时，`{ms_version}`可写为2.5.0。
> - `{mg_version}`表示MindSpore Golden Stick版本号，例如下载1.0.0版本MindSpore Golden Stick时，`{mg_version}`应写为1.0.0。

## 源码编译安装

下载[源码](https://gitee.com/mindspore/golden-stick)，下载后进入`golden_stick`目录。

```shell
git clone https://gitee.com/mindspore/golden-stick.git
cd golden-stick
bash build.sh
pip install output/mindspore_gs-{mg_version}-py3-none-any.whl
```

## 验证安装是否成功

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_gs
```
