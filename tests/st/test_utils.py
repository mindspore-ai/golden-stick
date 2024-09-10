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

import numpy as np
from mindspore import log as logger
from mindspore import load_checkpoint, Model
from mindspore.communication import get_rank
from mindspore.nn import Cell


# CI env
data_root = "/home/workspace/mindspore_dataset/"
ckpt_root = "/home/workspace/mindspore_ckpt/ckpt"
cur_path = os.path.split(os.path.realpath(__file__))[0]
model_zoo_path = os.path.join(cur_path, "../../../tests/models")


def check_network_contain_layer(network: Cell, layer_type, end_points: tuple = tuple()):
    """
    check if network contains some kind of type.
    """
    if not isinstance(network, Cell):
        return False
    if not issubclass(layer_type, Cell):
        return False
    for name, cell in network.name_cells().items():
        if isinstance(cell, layer_type):
            logger.info(f"{layer_type} exist in network, layer name: {name}")
            return True
        if end_points and isinstance(cell, end_points):
            continue
        if check_network_contain_layer(cell, layer_type, end_points):
            return True
    return False


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


def copy_file(src, dst):
    """
    Copy file named `file_name` from `from_` dir to `to_` dir. If `file_name` is a dir, copy all files under the dir. If
    `file_name` exist in `to_` dir, it will be deleted.
    """
    if not os.path.exists(src):
        raise ValueError("There is no file or path", src)
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    return os.system("cp -r {0} {1}".format(src, dst))


def exec_sed(file, src, dst):
    """ Replace `dst` string to `src` string in `file`. """
    cmd = 'sed -i "s/{0}/{1}/g" {2}'.format(src, dst, file)
    ret = os.system(cmd)
    return ret == 0, cmd


def process_check(cycle_time, cmd, wait_time=5):
    """ Wait and check process defined by `cmd` until it exit. """
    for i in range(cycle_time):
        time.sleep(wait_time)
        sub = subprocess.Popen(args="{}".format(cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
        stdout_data, _ = sub.communicate()
        if not stdout_data:
            logger.info("process execute success.")
            return True
        logger.info("process is running, please wait. time {}s.".format(i))
    logger.warning("process execute execute timeout.")
    return False


def load_distribut_checkpoint(config, ckpt_path, network):
    """load ckpt to multi card"""
    rank_id = get_rank() or 0
    bs = config.model.model_config.batch_size
    seq = config.model.model_config.seq_length
    if os.path.isdir(ckpt_path):
        for file in os.listdir(os.path.join(ckpt_path, f"rank_{rank_id}")):
            if not file.endswith(".ckpt"):
                continue
            ckpt_path = os.path.join(ckpt_path, f"rank_{rank_id}", file)
            model = Model(network)
            inputs = network.prepare_inputs_for_predict_layout(input_ids=np.ones([bs, seq], dtype=np.int32))
            model.infer_predict_layout(*inputs)
            break
    print(f'Loading ckpt :{ckpt_path}.')
    load_checkpoint(ckpt_path, network)
    return network


class TrainEvalConfig:
    """ A helper for replacing config into config file. """

    run_mode_key = "mode_name"
    epoch_size_key = "epoch_size"
    device_target_key = "device_target"
    batch_size_key = "batch_size"

    def __init__(self):
        """ Constructor of TrainEvalConfig. """
        self.config_list = {}

    @classmethod
    def run_mode_train_eval_config(cls, run_mode: str):
        """ create a run_mode TrainEvalConfig. """
        config = cls()
        config.set_config(TrainEvalConfig.run_mode_key, run_mode)
        return config

    def set_config(self, key: str, value: str):
        """ add config to config list """
        self.config_list[key] = value

    def get_run_mode(self):
        """ get run_mode from config list. """
        mode = self.config_list.get(TrainEvalConfig.run_mode_key)
        return mode if mode else "GRAPH"

    def apply(self, file: str):
        """ sed file with config list. """
        for key, value in self.config_list.items():
            ret, cmd = exec_sed(file, "^{}.*$".format(key), "{}: {}".format(key, value))
            if not ret:
                raise RuntimeError("Error occurred while executing {}".format(cmd))


def train_network(ori_model_path, model_name, model_suffix, config_name, algo_rpath, script_name,
                  config: TrainEvalConfig, ds_type="mnist", train_timeout_sec=300, continue_train=False,
                  pretrained=False, ckpt_path="", train_log_rpath=""):
    """
    Train a network.

    Args:
        ori_model_path (str): Model directory path to submodule models. Take LeNet as an example, the relative path of
          LeNet from this file is: '../models/official/cv/'.
        model_name (str): Name of directory, take LeNet as an example, name of directory is 'lenet'.
        model_suffix (str): Suffix for model path for unique directory.
        config_name (str): File name of config file. Take training LeNet network on MNIST dataset as an example, config
          file name is 'lenet_mnist_config.yaml'.
        algo_rpath (str): Relative path of algorithm from '$model_path/golden_stick'. Take SimQAT algorithm in LeNet as
          an example, algo_rpath is 'quantization/simqat'.
        script_name (str): File name of script under '$model_path/golden_stick/scripts'
        config (TrainEvalConfig): Train config to update config file.
        ds_type (str): Name of dataset, only support 'mnist' for MNIST dataset now!
        train_timeout_sec (int): Timeout in seconds, default 300.
        continue_train (bool): Is continue training.
        pretrained (bool): Is has pretrained original network.
        ckpt_path (str): Continue training or pretrained ckpt file path.
        train_log_rpath (str): Relative path of train log file from model_path.

    Returns:
        A string represents real working model directory. Take LeNet as an example, it returns '$cur_path/lenet'.
    """

    cur_path_ = os.path.dirname(os.path.abspath(__file__))
    copy_file(os.path.join(ori_model_path, model_name), os.path.join(cur_path_, model_name + model_suffix))
    model_path = os.path.join(cur_path_, model_name + model_suffix)
    assert os.path.exists(model_path)
    exec_path = os.path.join(model_path, "golden_stick", "scripts")
    algo_path = os.path.join(model_path, "golden_stick", algo_rpath)
    config_file = os.path.join(algo_path, config_name)
    assert os.path.exists(config_file)
    config.apply(config_file)
    ds_path = os.getenv("DATASET_PATH", None)
    if ds_path:
        if ds_type == "mnist":
            train_ds_path = os.path.join(ds_path, "mnist/train")
        elif ds_type == "cifar10":
            train_ds_path = os.path.join(ds_path, "cifar/cifar-10-batches-bin")
        else:
            raise NotImplementedError("ds_type not support {} now!".format(ds_type))
    else:
        if ds_type == "mnist":
            train_ds_path = os.path.join(data_root, "mnist/train")
        elif ds_type == "cifar10":
            train_ds_path = os.path.join(data_root, "cifar-10-batches-bin")
        else:
            raise NotImplementedError("ds_type not support {} now!".format(ds_type))
    assert os.path.exists(exec_path)
    assert os.path.exists(algo_path)
    assert os.path.exists(train_ds_path)
    assert os.path.exists(os.path.join(exec_path, script_name))
    if continue_train:
        assert os.path.exists(ckpt_path)
        exec_train_network_shell = "cd {}; bash {} {} {} {} PRETRAINED {}; cd -" \
            .format(exec_path, script_name, algo_path, config_file, train_ds_path, ckpt_path)
    elif pretrained:
        assert os.path.exists(ckpt_path)
        exec_train_network_shell = "cd {}; bash {} {} {} {} FP32 {}; cd -" \
            .format(exec_path, script_name, algo_path, config_file, train_ds_path, ckpt_path)
    else:
        exec_train_network_shell = "cd {}; bash {} {} {} {}; cd -" \
            .format(exec_path, script_name, algo_path, config_file, train_ds_path)
    logger.info(f"{'=' * 10} start training {model_name} in {config.get_run_mode()} mode {'=' * 10}")
    logger.info(f"Train cmd: {exec_train_network_shell}")
    ret = os.system(exec_train_network_shell)
    assert ret == 0
    cmd = "ps -ef | grep python | grep train.py | grep {} | grep -v grep".format(config_name)
    ret = process_check(train_timeout_sec, cmd, 1)
    if not ret:
        log_path = os.path.join(model_path, train_log_rpath)
        if os.path.exists(log_path):
            os.system("cat {}".format(log_path))
        else:
            os.system("echo {}".format("No train log file exist: " + log_path))
        assert ret
    logger.info(f"{'=' * 10} finish training {model_name} in {config.get_run_mode()} mode {'=' * 10}")
    return model_path


def eval_network(model_path, model_name, config_name, algo_rpath, script_name, ckpt_rpath, config: TrainEvalConfig,
                 ds_type="mnist", eval_timeout_sec=50, eval_log_file_name="log.txt",
                 acc_regex="=== {'Accuracy': ([0-9,.]*)} ===", train_log_rpath=""):
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
        config (TrainEvalConfig): Eval config to update config file.
        ds_type (str): Name of dataset, only support 'mnist' for MNIST dataset now!
        eval_timeout_sec (int): Timeout in seconds, default 50.
        eval_log_file_name (str): Name of log file.
        acc_regex (str): Regex str for matching acc-result.
        train_log_rpath (str): Relative path of train log file from model_path.

    Returns:
        A string represents real working model directory. Take LeNet as an example, it returns '$cur_path/lenet'.
    """

    assert os.path.exists(model_path)
    exec_path = os.path.join(model_path, "golden_stick", "scripts")
    algo_path = os.path.join(model_path, "golden_stick", algo_rpath)
    config_file = os.path.join(algo_path, config_name)
    assert os.path.exists(exec_path)
    assert os.path.exists(algo_path)
    assert os.path.exists(config_file)
    config.apply(config_file)
    ds_path = os.getenv("DATASET_PATH", None)
    if ds_path:
        if ds_type == "mnist":
            eval_ds_path = os.path.join(ds_path, "mnist/test")
        elif ds_type == "cifar10":
            eval_ds_path = os.path.join(ds_path, "cifar/cifar-10-verify-bin")
        else:
            raise NotImplementedError("ds_type not support {} now!".format(ds_type))
    else:
        if ds_type == "mnist":
            eval_ds_path = os.path.join(data_root, "mnist/test")
        elif ds_type == "cifar10":
            eval_ds_path = os.path.join(data_root, "cifar-10-verify-bin")
        else:
            raise NotImplementedError("ds_type not support {} now!".format(ds_type))

    ckpt_file = os.path.join(exec_path, ckpt_rpath)
    assert os.path.exists(os.path.join(exec_path, script_name))
    assert os.path.exists(eval_ds_path)
    if not os.path.exists(ckpt_file):
        log_path = os.path.join(model_path, train_log_rpath)
        if os.path.exists(log_path):
            os.system("cat {}".format(log_path))
        else:
            os.system("echo {}".format("No train log file exist: " + log_path))
        assert False
    exec_eval_network_shell = "cd {}; bash {} {} {} {} {}; cd -" \
        .format(exec_path, script_name, algo_path, config_file, eval_ds_path, ckpt_file)
    logger.info(f"{'=' * 10} start evaluating {model_name} in {config.get_run_mode()} mode {'=' * 10}")
    logger.info(f"Eval cmd: {exec_eval_network_shell}")
    ret = os.system(exec_eval_network_shell)
    assert ret == 0
    cmd = "ps -ef | grep python | grep eval.py | grep {} | grep -v grep".format(config_name)
    ret = process_check(eval_timeout_sec, cmd, 1)
    eval_log_file = os.path.join(exec_path, "eval", eval_log_file_name)
    if not ret:
        if os.path.exists(eval_log_file):
            os.system("cat {}".format(eval_log_file))
        else:
            os.system("echo {}".format("No eval log file exist: " + eval_log_file))
        assert ret
    logger.info(f"{'=' * 10} finish evaluating {model_name} in {config.get_run_mode()} mode {'=' * 10}")

    assert os.path.exists(eval_log_file)
    results = []
    with open(eval_log_file, "r") as file:
        for line in file.readlines():
            match_result = re.search(acc_regex, line)
            if match_result is not None:
                results.append(float(match_result.group(1)))
    assert len(results) == 1
    logger.info(f"{'=' * 10} LeNet {config.get_run_mode()} mode accuracy: {results[0]} {'=' * 10}")
    return results[0]


def relative_tolerance(data: np.ndarray, ground: np.ndarray):
    """Calculate relative tolerance."""
    diff = np.abs(data - ground)
    return diff / np.abs(ground + 1e-5)


def relative_tolerance_acceptable(data: np.ndarray, ground: np.ndarray, tolerance: float):
    """Calculate relative tolerance and check."""
    diff = relative_tolerance(data, ground)
    max_diff = np.max(diff)
    ret = max_diff < tolerance
    if not ret:
        logger.error(f"relative_tolerance: \r\n{diff}, \r\nmax: {max_diff}")
    return ret


def absolute_tolerance(data: np.ndarray, ground: np.ndarray):
    """Calculate relative tolerance."""
    return np.abs(data - ground)


def absolute_tolerance_acceptable(data: np.ndarray, ground: np.ndarray, tolerance: float):
    """Calculate relative tolerance and check."""
    diff = absolute_tolerance(data, ground)
    max_diff = np.max(diff)
    ret = max_diff < tolerance
    if not ret:
        logger.error(f"absolute_tolerance: \r\n{diff}, \r\nmax: {max_diff}")
    return ret
