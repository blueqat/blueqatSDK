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
"""This module provides a feature to flatten circuit operations by expanding multi-targets."""

import typing
from typing import Any, List

from .. import Circuit
from .. import gate as g


def flatten(c: Circuit) -> Circuit:
    """Expands slice and multiple targets into single target operations.

    This function normalizes the circuit so that each gate or measurement operation
    applies to explicit, un-sliced single qubits (or single pairs for two-qubit gates).

    Args:
        c (Circuit): The quantum circuit to flatten.

    Returns:
        Circuit: A new flattened Circuit object.

    Raises:
        ValueError: If an unexpected or unprocessable operation type is encountered.
    """
    n_qubits = c.n_qubits
    ops: List[g.Operation] = []
    
    for op in c.ops:
        if isinstance(op, (g.OneQubitGate, g.Reset)):
            ops.extend([
                op.create(t, op.params, None) 
                for t in op.target_iter(n_qubits)
            ])
            
        elif isinstance(op, g.TwoQubitGate):
            ops.extend([
                op.create(t, op.params, None)
                for t in op.control_target_iter(n_qubits)
            ])
            
        elif isinstance(op, g.Measurement):
            if op.key is None:
                ops.extend([
                    op.create(t, op.params, None) 
                    for t in op.target_iter(n_qubits)
                ])
            else:
                options: typing.Dict[str, Any] = {'key': op.key}
                if op.duplicated is not None:
                    options['duplicated'] = op.duplicated
                ops.append(
                    op.create(
                        tuple(t for t in op.target_iter(n_qubits)),
                        op.params, 
                        options
                    )
                )
        else:
            raise ValueError(f"Cannot process operation {op.lowername}.")
            
    return Circuit(n_qubits, ops)