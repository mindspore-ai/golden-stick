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
MindSpore golden stick base Custom op.
"""

import functools
import inspect
import enum
from typing import Union

from mindspore.ops import Custom, DataType, CustomRegOp
from mindspore import log as logger
from mindspore import context


class AttrValueType(enum.Enum):
    """Attr value type."""
    k_int = 'int'
    k_str = 'str'
    k_bool = 'bool'
    k_float = 'float'
    k_list_int = 'listInt'
    k_list_str = 'listStr'
    k_list_bool = 'listBool'
    k_list_float = 'listFloat'

    def __str__(self):
        """__str__"""
        return f"{self.name}"

    def value(self) -> str:
        """
        Return value of `AttrValueType`.

        Returns:
            An str as value of `AttrValueType`.
        """
        return self._value_

    @staticmethod
    def get_value_type(value: Union[int, str, bool, float, list]) -> str:
        """
        Get value type of input attribute value, return one of ["int", "str", "bool", "float", "listInt",
        "listStr", "listBool", "listFloat"].

        Args:
            value ([int, str, bool, float, list]): Input op attribute value.

        Returns:
            Op attribute value type, can be one of ["int", "str", "bool", "float", "listInt",
                "listStr", "listBool", "listFloat"]

        Raises:
            TypeError: If input value is not int, str, bool, float or list.
        """
        if isinstance(value, int):
            return AttrValueType.k_int.value()
        if isinstance(value, str):
            return AttrValueType.k_str.value()
        if isinstance(value, bool):
            return AttrValueType.k_bool.value()
        if isinstance(value, float):
            return AttrValueType.k_float.value()
        if isinstance(value, list):
            return AttrValueType.get_list_value_type(value)
        raise TypeError(f"For MindSpore Golden Stick, only support input attr value type in "
                        f"(int, str, bool, float, list), but got type {type(value).__name__}.")

    @staticmethod
    def get_list_value_type(value):
        """get_list_value_type"""
        if not value or not isinstance(value, list):
            raise TypeError(f"For MindSpore Golden Stick method 'get_list_value_type', only support non-empty list type"
                            f"input, but got {value}.")
        value_type = type(value[0])
        for single_value in value:
            if not isinstance(single_value, value_type):
                raise ValueError(f"For MindSpore Golden Stick method 'get_list_value_type', all items in value must be"
                                 f"the same type.")
        if isinstance(value[0], int):
            return AttrValueType.k_list_int.value()
        if isinstance(value[0], str):
            return AttrValueType.k_list_str.value()
        if isinstance(value[0], bool):
            return AttrValueType.k_list_bool.value()
        if isinstance(value[0], float):
            return AttrValueType.k_list_float.value()
        raise TypeError(f"For MindSpore Golden Stick, only support input list attr value type in "
                        f"(int, str, bool, float), but got type {type(value[0]).__name__}.")


class GSOpAttr:
    """
    Custom op attr.

    Args:
        attr_name (str): Custom op attribute name.
        attr_value ([int, str, bool, float, list]): Custom op attribute value.
        param_type (str): Custom op attribute parameter type, can be one of ["required", "optional"].
    """
    def __init__(self, attr_name: str, attr_value, param_type: str):
        self.name: str = attr_name
        self.value = attr_value
        self.type: str = AttrValueType.get_value_type(self.value)
        self.param_type = param_type

    def __str__(self):
        """output custom ops attr to string"""
        return f"Attr '{self.name}', value '{self.value}', type '{self.type}', param_type '{self.param_type}'"


def custom_op_attr_register(fn):
    """
    Custom op attributes register.
    Firstly, initialize an instance of Custom. Secondly, fetch custom ops attributes from function '__init__'
    automatically. Get attributes' name, value and param_type (if attribute has default value in '__init__', then its
    param_type is 'optional', otherwise is 'required'). Finally, call '_init_custom_op' to initialize OP.
    """

    @functools.wraps(fn)
    def deco(self, *args, **kwargs):
        """deco."""
        GSCustom.__init__(self)
        ops_sig = inspect.signature(fn)
        ops_params = ops_sig.parameters
        bound_args = ops_sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        arguments = bound_args.arguments
        del arguments['self']
        for name in arguments:
            value = arguments[name]
            param_type = "required" if type(ops_params[name]).__name__ != 'type' else 'optional'
            single_attr = GSOpAttr(name, value, param_type)
            self.custom_op_attr[name] = single_attr
            logger.info(f"Custom op {self._get_custom_op_name()} adding attr {single_attr}")
        logger.info(f"Custom op {self._get_custom_op_name()} begin to init op.")
        self._init_custom_op()
        logger.info(f"Custom op {self._get_custom_op_name()} init op success.")
        fn(self, *args, **kwargs)

    deco.decorated_func = fn
    return deco


class GSCustom(Custom):
    """
    Base class for MindSpore Golden Stick op.

    Examples:
        >>> class FooOps(GSCustom):
        >>>    @custom_op_attr_register
        >>>    def __init__(self, attribute_one, attribute_two=1):
        >>>        support_device = ["GPU"]
        >>>        self._check_support_device_target(support_device)
        >>>    def _infer_shape(self, x):
        >>>        return x
        >>>    def _infer_dtype(self, x):
        >>>        return x
        >>>    def _get_op_bprop(self):
        >>>        def bprop(x, out, dout):
        >>>            return zeros_like(x)
        >>>        return bprop
        >>>    def _get_op_input_names(self) -> (str,):
        >>>        return "x"
        >>>    def _get_op_output_names(self) -> (str,):
        >>>        return "output"
        >>>    def _get_op_dtype_formats(self) -> [[DataType]]:
        >>>        return [[DataType.F32_Default, DataType.F32_Default], [DataType.F16_Default, DataType.F16_Default]]
    """
    def __init__(self):
        """Initialize OP"""
        self.custom_op_attr: {str, GSOpAttr} = {}

    def _check_op_io_names(self, names: (str,)):
        """
        Check if input or output names are string.
        """
        for single_name in names:
            if not isinstance(single_name, str):
                raise TypeError(f"For {self._get_custom_op_name()}, op input name must be str, but got {single_name} "
                                f"which type is {type(single_name).__name__}")

    def _check_op_dtype_format(self, io_dtype: [[DataType]]):
        """
        Check if input dtype formats are list of list.
        """
        for one_format in io_dtype:
            if not isinstance(one_format, list):
                raise ValueError(f"For {self._get_custom_op_name()} op dtype format should be list, but got "
                                 f"{type(one_format).__name__}")

    def _get_op_reg_info(self):
        """
        Register info for op.
        A complete op reg info is show below:
            foo_op_info = CustomRegOp("foo_op_kernel") \
                .input(0, "x") \
                .output(0, "y") \
                .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
                .attr("num_bits", "required", "int", value=num_bits) \
                .target("GPU") \
                .get_op_info()
        """
        info_name = GSCustom._get_lower_name(self._get_custom_op_name()) + "_kernel"
        op_info = CustomRegOp(info_name)

        op_input_names = self._get_op_input_names()
        self._check_op_io_names(op_input_names)
        for index, single_name in enumerate(op_input_names):
            op_info = op_info.input(index, single_name)
        logger.info(f"Custom op {self._get_custom_op_name()} reg input names success")

        op_output_names = self._get_op_output_names()
        self._check_op_io_names(op_output_names)
        for index, single_name in enumerate(op_output_names):
            op_info = op_info.output(index, single_name)
        logger.info(f"Custom op {self._get_custom_op_name()} reg output names success")

        op_dtype_foramts = self._get_op_dtype_formats()
        self._check_op_dtype_format(op_dtype_foramts)
        for single_dtype_format in op_dtype_foramts:
            op_info = op_info.dtype_format(*single_dtype_format)
        logger.info(f"Custom op {self._get_custom_op_name()} reg dtype format success")

        for attr_name in self.custom_op_attr:
            gs_op_attr = self.custom_op_attr[attr_name]
            op_info = op_info.attr(gs_op_attr.name, gs_op_attr.param_type, gs_op_attr.type, value=gs_op_attr.value)
        logger.info(f"Custom op {self._get_custom_op_name()} reg attr success")

        op_info = op_info.target(context.get_context('device_target'))
        return op_info.get_op_info()

    def _get_forward_func(self) -> str:
        """
        Automatically generate farward func according to class name.

        Returns:
            Farward func, a string represent '{dir_path}/{file_name}:{func_name}'.
        """
        raise NotImplementedError

    @staticmethod
    def _get_lower_name(op_cls_name: str):
        """Get lower op cls name."""
        lower_name = ""
        for single_char in op_cls_name:
            if single_char.isupper():
                lower_name += "_" + single_char.lower()
            else:
                lower_name += single_char
        lower_name = lower_name.strip('_')
        return lower_name

    def _init_custom_op(self):
        """
        Init custom op, called in decorator 'custom_op_attr_register'.
        """
        forward_func = self._get_forward_func()
        logger.info(f"Custom op {self._get_custom_op_name()} get forward_func success.")
        op_bprop = self._get_op_bprop() if self._get_op_bprop() else None
        logger.info(f"Custom op {self._get_custom_op_name()} get op_bprop success.")
        op_reg_info = self._get_op_reg_info()
        logger.info(f"Custom op {self._get_custom_op_name()} get op_reg_info success.")
        super(GSCustom, self).__init__(func=forward_func, out_shape=self._infer_shape, out_dtype=self._infer_dtype,
                                       bprop=op_bprop, reg_info=op_reg_info, func_type='aot')

    def _check_support_device_target(self, device_target_list: [str]):
        """check_support_device_target"""
        device_target = context.get_context('device_target')
        if device_target not in device_target_list:
            error_str = f"MindSpore Golden Stick op not support in current device {device_target}."
            logger.error(error_str)
            raise ValueError(error_str)

    def _get_custom_op_name(self) -> str:
        """_get_custom_op_name"""
        return self.__class__.__name__

    def _get_custom_attr(self, attr_name: str):
        """_get_custom_attr"""
        if attr_name not in self.custom_op_attr.keys():
            raise RuntimeError(f"Custom op {self._get_custom_op_name()} do not have attr {attr_name}")
        return self.custom_op_attr.get(attr_name).value

    def _infer_shape(self, *inputs):
        """
        Infer output shape based on input shape.

        Args:
            inputs ((tuple(int), )): shapes of input tensors.

        Return:
            `(tuple(int), )`, shapes of output tensors.
        """
        raise NotImplementedError

    def _infer_dtype(self, *inputs):
        """
        Infer output dtype based on input dtype.

        Args:
            inputs ((:class:`mindspore.dtype`, )): data type of inputs.

        Return:
            (:class:`mindspore.dtype`, ), data type of outputs.
        """
        raise NotImplementedError

    def _get_op_bprop(self):
        """
        Bprop function of custom op.
        """
        return None

    def _get_op_input_names(self) -> (str,):
        """
        set_op_input_names.
        """
        raise NotImplementedError

    def _get_op_output_names(self) -> (str,):
        """
        set_op_output_names.
        """
        raise NotImplementedError

    def _get_op_dtype_formats(self) -> [[DataType]]:
        """
        set_op_dtype_format.
        """
        raise NotImplementedError
