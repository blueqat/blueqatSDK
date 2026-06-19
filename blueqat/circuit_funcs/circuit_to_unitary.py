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
"""This module provides a feature to convert a quantum circuit to a unitary matrix."""

import typing
from typing import Any

import numpy as np
from blueqat import Circuit


def circuit_to_unitary(circ: Circuit, *runargs: Any, **runkwargs: Any) -> np.ndarray:
    """Convert a quantum circuit into its corresponding unitary matrix representation.

    This function simulates the circuit for all computational basis states to construct
    the full unitary matrix.

    Args:
        circ (Circuit): The quantum circuit to be converted.
        *runargs: Positional arguments passed to circuit execution backend.
        **runkwargs: Keyword arguments passed to circuit execution backend.

    Returns:
        np.ndarray: The unitary matrix representing the total circuit operation.
    """
    runkwargs.setdefault('returns', 'statevector')
    runkwargs.setdefault('ignore_global', False)
    
    n_qubits = circ.n_qubits
    if n_qubits == 0:
        return np.array([[1.0 + 0.0j]], dtype=np.complex128)
        
    vecs = []
    # Loop over all computational basis states (from |00...0> to |11...1>)
    for i in range(1 << n_qubits):
        bitmask = tuple(k for k in range(n_qubits) if (1 << k) & i)
        
        # Initialize circuit with the correct size to maintain consistency
        c = Circuit(n_qubits)
        if bitmask:
            c.x[bitmask]
            
        c += circ
        vecs.append(c.run(*runargs, **runkwargs))
        
    # Column vectors correspond to outputs, transpose to form the matrix correctly
    return np.array(vecs).T