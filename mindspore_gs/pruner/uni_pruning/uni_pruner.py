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
"""
Implementation of UniPruning algorithm that zeroize model every n epochs according to UniPruning criterion.
Pruning mask from the last step and zeroed weights are used to get physically pruned model for inference.
"""
import os
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore import log as logger
from mindspore._checkparam import Validator
from mindspore.nn import Cell, Conv2d, Dense
from mindspore.train.callback import Callback

from ...comp_algo import CompAlgo
from .graph_analyzer import GraphAnalyzer
from .utils import do_mask, get_channel_importances, get_mask, prune_net, save_model_and_mask
from .unipruning_masked_layer import UniPruningMaskedConv2d, UniPruningMaskedDense
from ..ops import MaskedCell


class UniPrunerCallback(Callback):
    """
    UniPruning Callback that applies UniZeroing during training.
    Argument description is given in UniPruner class.

    Raises:
        TypeError: If `exp_name` is not string.
        TypeError: If `output_path` is not string.
        TypeError: If `input_size` is not tuple or list.
        TypeError: If `prune_flag` is not int.
        TypeError: If `frequency` is not int.
        TypeError: If `target_sparsity` is not float.
        TypeError: If `pruning_step` is not int.
        TypeError: If `filter_lower_threshold` is not int.
        TypeError: If `device_target` is not string.
        TypeError: If `rank` is not int.
    """

    def __init__(self, exp_name, output_path, input_size,
                 prune_flag, frequency, target_sparsity,
                 pruning_step, filter_lower_threshold,
                 device_target: GraphAnalyzer, rank):
        super().__init__()
        Validator.check_value_type("exp_name", exp_name, [str], self.__class__.__name__)
        Validator.check_value_type("output_path", output_path, [str], self.__class__.__name__)
        Validator.check_value_type("input_size", input_size, [tuple, list], self.__class__.__name__)
        Validator.check_value_type("prune_flag", prune_flag, [int], self.__class__.__name__)
        Validator.check_value_type("frequency", frequency, [int], self.__class__.__name__)
        Validator.check_value_type("target_sparsity", target_sparsity, [float], self.__class__.__name__)
        Validator.check_value_type("pruning_step", pruning_step, [int], self.__class__.__name__)
        Validator.check_value_type("filter_lower_threshold", filter_lower_threshold, [int], self.__class__.__name__)
        Validator.check_value_type("device_target", device_target, [str], self.__class__.__name__)
        Validator.check_value_type("rank", rank, [int], self.__class__.__name__)
        self.exp_name = exp_name
        self.output_path = os.path.join(output_path, exp_name)
        self.input_size = input_size
        self._frequency = frequency
        self._target_sparsity = target_sparsity
        self.pruning_step = pruning_step
        self.filter_lower_threshold = filter_lower_threshold
        self.mask = {}
        self.graph_anaylzer = None
        self.prune_flag = prune_flag
        self.rank = rank
        self.device_target = device_target
        if device_target == 'Ascend':
            self.save_model = True
        else:
            self.save_model = False
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def epoch_begin(self, run_context):
        if self.prune_flag == 0:
            return
        origin_args = run_context.original_args()
        cur_epoch_num = origin_args.cur_epoch_num
        if (cur_epoch_num - 1) % self._frequency == 0:
            logger.info(f'UniZeroing step before epoch {cur_epoch_num}')
            self.uni_zeroing(run_context)

    def epoch_end(self, run_context):
        if self.prune_flag == 0:
            return
        origin_args = run_context.original_args()
        cur_epoch_num = origin_args.cur_epoch_num
        if cur_epoch_num == origin_args.epoch_num and self.rank == 0:
            self.save_model = True
            logger.info('Final UniZeroing step rank', self.rank)
            self.uni_zeroing(run_context)

    def uni_zeroing(self, run_context):
        """
        UniZeroing step, consisting of:
            1. Computing channel importances as L2 norm of the channel,
             multiplied by L1 norm of the consecutive BatchNorm gamma parameter (if available).
            2. Grouping importances at each layer into groups of size UniPrunerCallback.pruning_step.
            3. At each layer compute channel group criterion as highest median in layer / median(group).
            4. Choose groups with highest criterion until reached target sparsity.
             Make mask as dict {layer: chosen channels}.
            5. Zeroize chosen channels.
            6. Save zeroed weights and pruning mask.
        """
        # compute channel importances and get pruning mask
        norms = get_channel_importances(self.graph_anaylzer.groups, self.filter_lower_threshold)
        self.mask = get_mask(self.graph_anaylzer.groups, norms, self.pruning_step,
                             self.filter_lower_threshold, self._target_sparsity)
        # apply pruning mask -> zero model
        do_mask(self.graph_anaylzer.groups, self.mask)
        origin_args = run_context.original_args()
        net: Cell = origin_args.network
        cur_epoch_num = origin_args.cur_epoch_num
        save_model_and_mask(net, self.output_path, f'{self.exp_name}_zeroed_rank{self.rank}',
                            cur_epoch_num, self.input_size, self.device_target, self.save_model,
                            self.mask)
        logger.info(f'UniZeroing ended rank{self.rank}')


class UniPruner(CompAlgo):
    """
    Derived class of `CompAlgo`. Base class of UniPruning algorithm.

    Args:
        config (dict): store attributes for quantization aware training, keys are attribute names,
            values are attribute values. supported attribute are listed below:

            - prune_flag (int): If set to 1, UniPruning is enabled. If set to 0, UniPruning is disabled.
              Default: 1
            - target_sparsity (float): Target compression rate of the pruned model.
            - frequency (int): Each frequency epochs model would be zeroed under UniPruning algorithm.
              Default: 9
            - pruning_step (int): The number of channels which would be zeroed/pruned as a single unit.
              Default: 32
            - filter_lower_threshold (int): The minimal number of channels in each layer in a pruned model.
              Default: 32
            - input_size (Union(tuple, list)): The shape of input tensor when exporting model into .MINDIR and .AIR.
              Default: (16, 3, 224, 224)
            - exp_name (str): Experiment name. Checkpoints and masks would be named as exp_name_epoch.
            - output_path (str): Path into which zeroed/pruned checkpoints and pruning masks would be saved.
            - rank (int): Local rank of the model when used in distributed training.
              Default: 0
            - device_target: Device on which experiment is performed (Ascend, GPU).
              Default: Ascend

    Raises:
        TypeError: If `config` is not dict.

    Supported Platforms:
        ``GPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore_gs.pruner.uni_pruning import UniPruner
        >>> ## 1) Fill pruning configuration dictionary
        >>> cfg = {
        >>>     "exp_name": config.exp_name,
        >>>     "frequency": config.retrain_epochs,
        >>>     "target_sparsity": 1 - config.prune_rate,
        >>>     "pruning_step": config.pruning_step,
        >>>     "filter_lower_threshold": config.filter_lower_threshold,
        >>>     "input_size": input_size,
        >>>     "output_path": config.save_checkpoint_path,
        >>>     "prune_flag": config.prune_flag,
        >>>     "rank": config.device_id,
        >>>     "device_target": config.device_target
        >>> }
        >>> ## 2) Create network for pruning
        >>> net = NetToPrune()
        >>> ## 3) Define Pruning algorithm
        >>> algo = UniPruner(cfg)
        >>> ## 4) Apply pruning algorithm to origin network
        >>> algo.apply(net)
        >>> ## 5) Train network for several epochs as usual
        >>> ## 6) Load pruning mask and checkpoint which have been generated during training
        >>> if config.mask_path is not None and os.path.exists(config.mask_path):
        >>> with open(config.mask_path, 'r', encoding='utf8') as json_fp:
        >>>     mask = json.load(json_fp)
        >>>     tag = 'pruned'
        >>> else:
        >>>     mask = None
        >>>     tag = 'original'
        >>> ms.load_checkpoint(config.checkpoint_file_path, net)
        >>> ## 7) Make pruning, save pruned network
        >>> algo.convert(net, mask, config, tag)
    """

    def __init__(self, config=None):
        super().__init__(config)
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._callback = UniPrunerCallback(exp_name=config["exp_name"],
                                           output_path=config["output_path"],
                                           input_size=config["input_size"],
                                           frequency=config["frequency"],
                                           target_sparsity=config["target_sparsity"],
                                           pruning_step=config["pruning_step"],
                                           filter_lower_threshold=config["filter_lower_threshold"],
                                           prune_flag=config["prune_flag"],
                                           rank=config["rank"],
                                           device_target=config["device_target"])
        self.graph_anaylzer = None

    def apply(self, network: Cell) -> Cell:
        """
        Analyze network computational graph.
        Currently supported networks:
            - Resnet-like models
            - VGG-like models
            - LeNet models

        Args:
            network (Cell): network for pruning.

        Raises:
            ValueError: If network type is not supported by graph analyzer.
        """
        fake_input = Tensor(np.ones(self._callback.input_size).astype(np.float32))
        self.graph_anaylzer = GraphAnalyzer(network, fake_input)
        self._callback.graph_anaylzer = self.graph_anaylzer

        for name, layer in network.cells_and_names():
            if isinstance(layer, Conv2d):
                origin_weight_name = layer.weight.name
                origin_bias_name = "bias"
                if layer.has_bias:
                    origin_bias_name = layer.bias.name
                new_cell = UniPruningMaskedConv2d(layer)
                network.insert_child_to_cell(name, new_cell)
                new_cell.handler.weight.name = "masked_{}".format(origin_weight_name)
                new_cell.in_mask.name = "{}.in_mask".format(name)
                new_cell.out_mask.name = "{}.out_mask".format(name)
                if layer.has_bias:
                    new_cell.handler.bias.name = "masked_{}".format(origin_bias_name)
            if isinstance(layer, Dense):
                origin_weight_name = layer.weight.name
                origin_bias_name = "bias"
                if layer.has_bias:
                    origin_bias_name = layer.bias.name
                new_cell = UniPruningMaskedDense(layer)
                network.insert_child_to_cell(name, new_cell)
                new_cell.handler.weight.name = "masked_{}".format(origin_weight_name)
                new_cell.in_mask.name = "{}.in_mask".format(name)
                new_cell.out_mask.name = "{}.out_mask".format(name)
                if layer.has_bias:
                    new_cell.handler.bias.name = "masked_{}".format(origin_bias_name)
        return network

    def prune_by_mask(self, net: Cell, mask, args, tag):
        """
        Prune network (optional) according to mask, save weights as .ckpt, model as .MINDIR and .AIR

        Args:
            net (Cell): network.
            mask (dict): pruning mask where layer is key and value is an array of channels to be pruned.
            args (dict): config arguments.
            tag (str): postfix for checkpoint's name.
                Default: original

        Raises:
            TypeError: If `mask` is not dict.
            TypeError: If `args` is not dict.
            TypeError: If `tag` is not string.
        """
        if mask is not None:
            Validator.check_value_type("mask", mask, [dict], self.__class__.__name__)
        Validator.check_value_type("tag", tag, [str], self.__class__.__name__)
        if mask is not None:
            prune_net(self.graph_anaylzer.groups, mask)
        save_model_and_mask(net, self._callback.output_path, f'{args.exp_name}_{tag}_{self._callback.rank}',
                            args.epoch_size, self._callback.input_size, args.device_target,
                            export_air=True)

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """prune network using checkpoint with zeroed weights and pruning mask saved"""
        if not isinstance(net_opt, Cell):
            raise TypeError(
                f'The parameter `net_opt` must be isinstance of Cell, but got {type(net_opt)}.')
        if not isinstance(ckpt_path, str):
            raise TypeError(
                f'The parameter `ckpt_path` must be isinstance of str, but got {type(ckpt_path)}.')
        real_path = os.path.realpath(ckpt_path)
        if ckpt_path != "":
            if os.path.isfile(real_path):
                param_dict = load_checkpoint(ckpt_path)
                load_param_into_net(net_opt, param_dict)
            else:
                raise ValueError(
                    f'The parameter `ckpt_path` can only be empty or a valid file, but got {real_path}.')
        for name, layer in net_opt.cells_and_names():
            if not isinstance(layer, MaskedCell):
                continue
            net_opt.insert_child_to_cell(name, layer.prune())
        return net_opt

    def callbacks(self, *args, **kwargs) -> [Callback]:
        """get UniPruner callback"""
        if self.graph_anaylzer is None:
            raise RuntimeError("Please call the apply interface first.")
        return [self._callback]
