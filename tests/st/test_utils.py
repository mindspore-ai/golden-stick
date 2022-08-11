# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Golden stick ST utils: check AlgoConfig with config"""

import os
import shutil
import subprocess
import time
import re
from mindspore import log as logger

# CI env
rank_table_path = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"
data_root = "/home/workspace/mindspore_dataset/"
ckpt_root = "/home/workspace/mindspore_dataset/checkpoint"
cur_path = os.path.split(os.path.realpath(__file__))[0]
model_zoo_path = os.path.join(cur_path, "../../../tests/models")


def qat_config_compare(algo_cfg, target: dict):
    """
    Compare a config object to a dict.
    Config object could be a SimulatedQuantizationConfig, a LearnedStepSizeQuantizationConfig or a SlbQuantConfig.
    """
    if not algo_cfg:
        logger.error("Config is none")
        return False
    if not isinstance(target, dict):
        logger.error("Target is not a dict")
        return False
    for key, value in target.items():
        act_key = "act_" + key
        weight_key = "weight_" + key
        if isinstance(value, (tuple, list)):
            if not hasattr(algo_cfg, act_key):
                logger.error("Config has no attribute " + act_key)
                return False
            if not hasattr(algo_cfg, weight_key):
                logger.error("Config has no attribute " + weight_key)
                return False
            if len(value) != 2:
                logger.error(f"Target value error, value({value}) of {key} should has two elements")
                return False
            act_value = getattr(algo_cfg, act_key)
            weight_value = getattr(algo_cfg, weight_key)
            if act_value != value[0]:
                logger.error(f"Config's {act_key}({act_value}) is not equal to target({value[0]})")
                return False
            if weight_value != value[1]:
                logger.error(f"Config's {weight_key}({weight_value}) is not equal to target({value[1]})")
                return False
        else:
            has_key = hasattr(algo_cfg, key)
            has_act_key = hasattr(algo_cfg, act_key)
            has_weight_key = hasattr(algo_cfg, weight_key)
            if has_act_key and not has_weight_key:
                logger.error(f"Config is invalid, has attribute {act_key} but no attribute {weight_key}")
                return False
            if not has_act_key and has_weight_key:
                logger.error(f"Config is invalid, has attribute {weight_key} but no attribute {act_key}")
                return False
            if not has_key and not has_act_key:
                logger.error("Config has no attribute " + key)
                return False
            if has_key and has_act_key:
                logger.error(f"Config has attribute {key} and {act_key}, {weight_key}")
                return False
            if has_key:
                cfg_value = getattr(algo_cfg, key)
                if cfg_value != value:
                    logger.error(f"Config's {key}({cfg_value}) is not equal to target({value})")
                    return False
            else:
                act_value = getattr(algo_cfg, act_key)
                weight_value = getattr(algo_cfg, weight_key)
                if act_value != value:
                    logger.error(f"Config's {act_key}({act_value}) is not equal to target({value})")
                    return False
                if weight_value != value:
                    logger.error(f"Config's {weight_key}({weight_value}) is not equal to target({value})")
                    return False
    return True


def copy_files(from_, to_, file_name):
    """
    Copy file named `file_name` from `from_` dir to `to_` dir. If `file_name` is a dir, copy all files under the dir. If
    `file_name` exist in `to_` dir, it will be deleted.
    """
    if not os.path.exists(os.path.join(from_, file_name)):
        raise ValueError("There is no file or path", os.path.join(from_, file_name))
    if os.path.exists(os.path.join(to_, file_name)):
        shutil.rmtree(os.path.join(to_, file_name))
    return os.system("cp -r {0} {1}".format(os.path.join(from_, file_name), to_))


def exec_sed(file, src, dst):
    """ Replace `dst` string to `src` string in `file`. """
    ret = os.system('sed -i "s#{0}#{1}#g" {2}'.format(src, dst, file))
    return ret == 0


def process_check(cycle_time, cmd, wait_time=5):
    """ Wait and check process defined by `cmd` until it exit. """
    for i in range(cycle_time):
        time.sleep(wait_time)
        sub = subprocess.Popen(args="{}".format(cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
        stdout_data, _ = sub.communicate()
        if sub.returncode != 0:
            logger.warning("process execute failed: {}.".format(sub.returncode))
            return False
        if not stdout_data:
            logger.info("process execute success.")
            return True
        logger.info("process is running, please wait {}".format(i))
    logger.warning("process execute execute timeout.")
    return False


def train_network(ori_model_path, model_name, config_name, algo_rpath, script_name, run_mode="GRAPH", ds_type="mnist",
                  train_timeout_sec=300):
    """
    Train a network.

    Args:
        ori_model_path (str): Model directory path to submodule models. Take LeNet as an example, the relative path of
          LeNet from this file is: '../models/official/cv/'.
        model_name (str): Name of directory, take LeNet as an example, name of directory is 'lenet'.
        config_name (str): File name of config file. Take training LeNet network on MNIST dataset as an example, config
          file name is 'lenet_mnist_config.yaml'.
        algo_rpath (str): Relative path of algorithm from '$model_path/golden_stick'. Take SimQAT algorithm in LeNet as
          an example, algo_rpath is 'quantization/simqat'.
        script_name (str): File name of script under '$model_path/golden_stick/scripts'
        run_mode (str): Training mode, "GRAPH" for mindspore.context.GRAPH_MODE, "PYNATIVE" for
          mindspore.context.PYNATIVE_MODE.
        ds_type (str): Name of dataset, only support 'mnist' for MNIST dataset now!
        train_timeout_sec (int): Timeout in seconds, default 300.

    Returns:
        A string represents real working model directory. Take LeNet as an example, it returns '$cur_path/lenet'.
    """

    if ds_type == "mnist":
        ds_sub_dir = "mnist/train"
    else:
        raise NotImplementedError("ds_type only support mnist now!")

    cur_path_ = os.path.dirname(os.path.abspath(__file__))
    copy_files(ori_model_path, cur_path_, model_name)
    model_path = os.path.join(cur_path_, model_name)
    assert os.path.exists(model_path)
    exec_path = os.path.join(model_path, "golden_stick", "scripts")
    algo_path = os.path.join(model_path, "golden_stick", algo_rpath)
    config_file = os.path.join(algo_path, config_name)
    assert os.path.exists(config_file)
    exec_sed(config_file, "GRAPH", run_mode)
    exec_sed(config_file, "PYNATIVE", run_mode)
    ds_path = os.getenv("DATASET_PATH", None)
    if ds_path:
        train_ds_path = os.path.join(ds_path, ds_sub_dir)
    else:
        train_ds_path = os.path.join(data_root, ds_sub_dir)
    assert os.path.exists(exec_path)
    assert os.path.exists(algo_path)
    assert os.path.exists(train_ds_path)
    assert os.path.exists(os.path.join(exec_path, script_name))
    exec_train_network_shell = "cd {}; bash {} {} {} {}; cd -" \
        .format(exec_path, script_name, algo_path, config_file, train_ds_path)
    print("=" * 10, "start training {} in {} mode".format(model_name, run_mode), "=" * 10, flush=True)
    print("Train cmd: {}".format(exec_train_network_shell), flush=True)
    ret = os.system(exec_train_network_shell)
    assert ret == 0
    cmd = "ps -ef | grep python | grep train.py | grep {} | grep -v grep".format(config_name)
    process_check(train_timeout_sec, cmd, 1)
    print("=" * 10, "finish training {} in {} mode.".format(model_name, run_mode), "=" * 10, flush=True)
    return model_path


def eval_network(model_path, model_name, config_name, algo_rpath, script_name, ckpt_rpath, run_mode="GRAPH",
                 ds_type="mnist", eval_timeout_sec=50):
    """
    Eval a network.

    Args:
        model_path (str): Real working model path. Usually feed return value of `train_network` to this parameter.
        model_name (str): Name of directory, take LeNet as an example, name of directory is 'lenet'.
        config_name (str): File name of config file. Take training LeNet network on MNIST dataset as an example, config
          file name is 'lenet_mnist_config.yaml'.
        algo_rpath (str): Relative path of algorithm from '$model_path/golden_stick'. Take SimQAT algorithm in LeNet as
          an example, algo_rpath is 'quantization/simqat'.
        script_name (str): File name of script under '$model_path/golden_stick/scripts'
        ckpt_rpath (str): Relative path of checkpoint file from '$model_path/golden_stick/scripts'
        run_mode (str): Training mode, "GRAPH" for mindspore.context.GRAPH_MODE, "PYNATIVE" for
          mindspore.context.PYNATIVE_MODE.
        ds_type (str): Name of dataset, only support 'mnist' for MNIST dataset now!
        eval_timeout_sec (int): Timeout in seconds, default 50.

    Returns:
        A string represents real working model directory. Take LeNet as an example, it returns '$cur_path/lenet'.
    """

    if ds_type == "mnist":
        ds_sub_dir = "mnist/test"
    else:
        raise NotImplementedError("ds_type only support mnist now!")

    assert os.path.exists(model_path)
    exec_path = os.path.join(model_path, "golden_stick", "scripts")
    algo_path = os.path.join(model_path, "golden_stick", algo_rpath)
    config_file = os.path.join(algo_path, config_name)
    assert os.path.exists(config_file)
    exec_sed(config_file, "GRAPH", run_mode)
    exec_sed(config_file, "PYNATIVE", run_mode)
    ds_path = os.getenv("DATASET_PATH", None)
    if ds_path:
        eval_ds_path = os.path.join(ds_path, ds_sub_dir)
    else:
        eval_ds_path = os.path.join(data_root, ds_sub_dir)

    ckpt_file = os.path.join(exec_path, ckpt_rpath)
    assert os.path.exists(exec_path)
    assert os.path.exists(algo_path)
    assert os.path.exists(os.path.join(exec_path, script_name))
    assert os.path.exists(eval_ds_path)
    assert os.path.exists(ckpt_file)
    exec_eval_network_shell = "cd {}; bash {} {} {} {} {}; cd -" \
        .format(exec_path, script_name, algo_path, config_file, eval_ds_path, ckpt_file)
    print("=" * 10, "start evaling {} in {} mode".format(model_name, run_mode), "=" * 10, flush=True)
    print("Eval cmd: {}".format(exec_eval_network_shell), flush=True)
    ret = os.system(exec_eval_network_shell)
    assert ret == 0
    cmd = "ps -ef | grep python | grep eval.py | grep {} | grep -v grep".format(config_name)
    process_check(eval_timeout_sec, cmd, 1)
    print("=" * 10, "finish evaling {} in {} mode.".format(model_name, run_mode), "=" * 10, flush=True)

    eval_log_file = os.path.join(exec_path, "eval", "log.txt")
    assert os.path.exists(eval_log_file)
    results = []
    with open(eval_log_file, "r") as file:
        for line in file.readlines():
            match_result = re.search("=== {'Accuracy': ([0-9,.]*)} ===", line)
            if match_result is not None:
                results.append(float(match_result.group(1)))
    assert len(results) == 1
    print("=" * 10, "LeNet {} mode accuracy: {}".format(run_mode, results[0]), "=" * 10, flush=True)
    return results[0]
