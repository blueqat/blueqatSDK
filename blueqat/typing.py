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
"""This module provides for type hinting within Blueqat.
Updated in 2026 to support PyTorch tensor-based qubit indexing and advanced static analysis.
"""

import typing
from typing import Generic, TypeVar, Union, Any

import torch

if typing.TYPE_CHECKING:
    from blueqat import Circuit

T = TypeVar('T')
C = TypeVar('C')

# 2026年型拡張: 従来の組み込み型に加え、PyTorchのテンソルを直接ターゲット（量子ビット指定）に受けるケースに対応
Targets = Union[int, slice, tuple, list, torch.Tensor]


class GeneralCircuitOperation(Generic[C, T]):
    """Type definition of dynamic method for quantum operations."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call method for parameterized gates (e.g., rx(0.5)).
        
        Returns a gate instance that can then accept target qubits via __getitem__.
        """
        ...

    def __getitem__(self, targets: Targets) -> C:
        """Getitem method for indexing target qubits (e.g., [0] or [1, 2]).
        
        Applies the operation to the circuit and returns the circuit instance 
        to enable method chaining.
        """
        ...


# Python 3.14環境下で、さらに `CircuitOperation[Circuit]` とラップして
# 継承できるようにするため、型エイリアスとして正しく再定義します。
# 呼び出し側が第一引数（C）をさらにカスタマイズできるように TypeVar を残します。
_C = TypeVar('_C')
CircuitOperation = GeneralCircuitOperation[_C, Any]