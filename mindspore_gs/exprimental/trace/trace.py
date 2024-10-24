# Copyright 2024 Huawei Technologies Co., Ltd
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
"""trace network."""

import functools
from functools import partial
import inspect
from types import FunctionType, CodeType

from typing import Tuple, List, Any
from mindspore.nn import Cell
import mindspore.ops.functional as F
from mindspore import context
from mindspore_gs.ptq.processor import Processor
from llama2 import create_llama


class TraceGraph:
    """TraceGraph"""
    def __init__(self, name: str):
        self.name = name
        self.type = None
        self.handler = None
        self.status = 0
        self.nodes: [TraceGraph] = []

    def before_execute(self, cell):
        """before_execute"""
        if self.status != 0:
            raise RuntimeError(f"Current trace graph status is {self.status}, is not ready for before_execute.")
        self.handler = cell
        self.type = type(cell)
        self.status = 1

    def append_node(self, node):
        """append_node"""
        if self.status != 1:
            raise RuntimeError(f"Current trace graph status is {self.status}, is not ready for append_node.")
        self.nodes.append(node)

    def after_execute(self):
        """after_execute"""
        if self.status != 1:
            raise RuntimeError(f"Current trace graph status is {self.status}, is not ready for after_execute.")
        self.status = 2

    def dump(self, indent=0):
        """dump"""
        print(f"{' ' * indent}{self.name} {self.type}", flush=True)
        for node in self.nodes:
            node.dump(indent + 4)


class Node:
    """Node"""
    def __init__(self, targets, kind: str, op, args, kwargs):
        if isinstance(targets, (tuple, list, set)):
            self.targets = list(targets)
        else:
            self.targets = [targets]
        self.kind = kind
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        res = ""
        # target:
        if self.targets:
            for target in self.targets:
                if res:
                    res += f" %{target}"
                else:
                    res += f"%{target}"
            res += " = "
        # args:
        args_str = ""
        for index, arg in enumerate(self.args):
            arg_str = arg.targets[0] if isinstance(arg, Node) else arg
            if index == 0:
                args_str += f"{arg_str}"
            else:
                args_str += f", {arg_str}"
        # function:
        if self.kind == 'return':
            res += f"return {args_str}"
        else:
            res += f"{self.kind}.{self.op}({args_str})"
        return res


class Graph:
    """Graph"""
    def __init__(self):
        self.nodes = []
        self.target_mgr = {}

    def append(self, node: Node):
        """append"""
        for i, target in enumerate(node.targets):
            cur_target_num = self.target_mgr.get(target, 0)
            if cur_target_num == 0:
                self.target_mgr[target] = 1
            else:
                self.target_mgr[target] = cur_target_num + 1
                node.targets[i] = f"{target}_{cur_target_num}"
        self.nodes.append(node)

    def __str__(self):
        code = ""
        for node in self.nodes:
            code += str(node) + "\r\n"
        return code


graph = Graph()


@functools.wraps(Cell.construct)
def call_construct(ori_fn, *args, **kwargs):
    """call_construct"""
    fn_for_analysis = inspect.unwrap(ori_fn)
    name = str(fn_for_analysis).split()[2].split('.')[0]
    proxy = Node(name.lower(), 'call_cell', name, args, kwargs)
    graph.append(proxy)
    return proxy


def create_placeholder(name, sig):
    """create_placeholder"""
    if name[0] == "*":
        default = ()
    else:
        param = sig.parameters[name]
        default = () if param.default is inspect.Parameter.empty else (param.default,)
    proxy = Node(name.lower(), 'placeholder', name, list(default), {})
    graph.append(proxy)
    return proxy


HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    """_patch_function"""
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    if hasattr(co, "co_qualname"):
        # Python-3.11+ code signature
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_qualname,  # type: ignore[attr-defined]
            co.co_firstlineno,
            co.co_lnotab,
            co.co_exceptiontable,  # type: ignore[attr-defined]
            co.co_freevars,
            co.co_cellvars,
        )
    elif hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    else:
        co_args = (
            nargs,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables


def trace_inputs(network: Cell):
    """trace_inputs"""
    fn_for_analysis = inspect.unwrap(network.construct)
    co = fn_for_analysis.__code__
    total_args = co.co_argcount + co.co_kwonlyargcount
    names_iter = iter(co.co_varnames)
    args: List[Any] = []
    if total_args == 0:
        raise RuntimeError("``self`` argument cannot be part of *args expansion!")
    skip_arg_idx = 1
    next(names_iter)  # skip self
    args.append(network)

    sig = inspect.signature(fn_for_analysis)

    arg_names = [next(names_iter) for _ in range(skip_arg_idx, total_args)]
    args.extend(create_placeholder(names, sig) for names in arg_names)
    network.construct = _patch_function(network.construct, len(args))
    return network.construct, args


def trace_cell(network: Cell):
    """trace_cell"""
    class AddHook(Processor):
        """AddHook"""
        def __init__(self):
            self.root = None
            self.stack: [TraceGraph] = []

        @staticmethod
        def _is_leaf_cell(cell) -> bool:
            if cell.__module__.startswith("mindspore.nn"):
                return True
            if any(name in type(cell).__name__ for name in
                   ['Embedding', 'FeedForward', 'RMSNorm', 'Attention', 'Linear']):
                return True
            return False

        def process_cell(self, _, cell: Cell) -> Tuple[Cell, bool]:
            if AddHook._is_leaf_cell(cell):
                cell.construct = partial(call_construct, cell.construct)
            return cell, False

    add_hooker = AddHook()
    add_hooker.process(network)


def trace_functional():
    """trace_functional"""
    funcs = ('reshape', 'shape', 'cast')
    for func in funcs:
        def patched_func(func, *args, **kwargs):
            proxy = Node(func.lower(), 'call_functional', func, args, kwargs)
            graph.append(proxy)
            return proxy
        patched_func.__name__ = func
        setattr(F, func, partial(patched_func, func))


reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and_': '{} & {}',
    'or_': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]',
    'matmul': '{} @ {}',
}


magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)


def trace_magic_methods():
    """trace_magic_methods"""
    for method in magic_methods:
        def patched_method(method, *args, **kwargs):
            proxy = Node(method.lower(), 'call_functional', method, args, kwargs)
            graph.append(proxy)
            return proxy
        patched_method.__name__ = method
        as_magic = f'__{method.strip("_")}__'
        setattr(Node, as_magic, partial(patched_method, method))


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    network = create_llama(5)
    trace_magic_methods()
    trace_functional()
    trace_cell(network)
    fn, args = trace_inputs(network)
    output = fn(*args)
    return_node = Node([], 'return', None, [output], {})
    graph.append(return_node)
    print(graph)
