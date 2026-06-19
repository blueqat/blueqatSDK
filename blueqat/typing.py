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
"""This module provides for type hinting within Blueqat."""

import typing
from typing import Generic, TypeVar, Union, Any

if typing.TYPE_CHECKING:
    from blueqat import Circuit

T = TypeVar('T')
C = TypeVar('C')

# Targets can be an integer, a slice (e.g. 0:3), or a tuple of multiple targets
Targets = Union[int, slice, tuple]


class GeneralCircuitOperation(Generic[C, T]):
    """Type definition of dynamic method for quantum operations."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Call method for parameterized gates (e.g., rx(0.5))."""
        ...

    def __getitem__(self, targets: Targets) -> C:
        """Getitem method for indexing target qubits (e.g., [0] or [1, 2])."""
        ...


# Type alias for standard circuit operations mapped to the Circuit object
CircuitOperation = GeneralCircuitOperation['Circuit', T]