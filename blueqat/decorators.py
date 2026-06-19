# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decorators for Blueqat Circuit Extensions."""

from keyword import iskeyword
from typing import Callable, Union, Optional

from .circuit import BlueqatGlobalSetting


def circuitmacro(func: Union[Callable, str, None] = None,
                 *,
                 allow_overwrite: bool = True) -> Callable:
    """@circuitmacro decorator.

    Typical usage:
    Case 1: no arguments

        @circuitmacro
        def egg(c):
            ...

    equivalent to this:

        def egg(c):
            ...
        BlueqatGlobalSetting.register_macro('egg', egg, allow_overwrite=True)


    Case 2: with name:

        @circuitmacro('bacon')
        def egg(c):
            ...

    is equivalent with

        def egg(c):
            ...
        BlueqatGlobalSetting.register_macro('bacon', egg, allow_overwrite=True)

    Case 3: with allow_overwrite keyword argument

        @circuitmacro(allow_overwrite=False)
        def egg(c):
            ...

    or

        @circuitmacro('bacon', allow_overwrite=False)
        def bacon(c):
            ...

    call BlueqatGlobalSetting.register_macro with allow_overwrite=False.

    Please note that `allow_overwrite=True` is default behavior.
    It is convenient for interactive environments like Jupyter Notebook.
    However, if you're a library developer, using `allow_overwrite=False` is highly recommended.
    """
    if callable(func):
        name = func.__name__
        if not name.isidentifier() or iskeyword(name):
            raise ValueError(
                f'Function name {name} is not a valid macro name.')
        BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
        return func

    if isinstance(func, str):
        name = func

        def _wrapper1(f: Callable) -> Callable:
            BlueqatGlobalSetting.register_macro(name, f, allow_overwrite)
            return f

        return _wrapper1

    if func is None:
        def _wrapper2(f: Callable) -> Callable:
            name_inner = f.__name__
            if not name_inner.isidentifier() or iskeyword(name_inner):
                raise ValueError(
                    f'Function name {name_inner} is not a valid macro name.')
            BlueqatGlobalSetting.register_macro(name_inner, f, allow_overwrite)
            return f

        return _wrapper2

    raise TypeError('Invalid type for first argument.')