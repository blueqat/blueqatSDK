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
"""The 'eo' backend: transpile a logical Circuit into exchange pulses.

    import blueqat.eo  # registers the backend
    physical = Circuit(2).h[0].cx[0, 1].run(backend='eo')

Logical qubit i is encoded in physical spins 3i, 3i+1, 3i+2, and the output
is an ordinary Circuit containing only `exch` pulses, runnable on any
simulation backend. All logical gates are exact up to global phase.
"""

from typing import Any, List

from ..backends.backendbase import Backend, register_backend
from ..circuit import Circuit
from ..gate import Operation
from . import sequences


class EOTranspiler(Backend):
    """Transpiler backend converting logical circuits to exchange pulses."""

    _FIXED = {
        'x': sequences.x_sequence,
        'y': sequences.y_sequence,
        'z': sequences.z_sequence,
        'h': sequences.h_sequence,
        's': sequences.s_sequence,
        'sdg': sequences.sdg_sequence,
        't': sequences.t_sequence,
        'tdg': sequences.tdg_sequence,
    }
    _ROTATIONS = {
        'rz': sequences.rz_sequence,
        'phase': sequences.rz_sequence,   # equal to rz up to global phase
        'rx': sequences.rx_sequence,
        'ry': sequences.ry_sequence,
    }

    def run(self, gates: List[Operation], n_qubits: int, *args: Any,
            **kwargs: Any) -> Circuit:
        pulses: List[sequences.Pulse] = []
        for gate in gates:
            name = gate.lowername
            if name in ('i', 'barrier'):
                continue
            if name in self._FIXED:
                for t in gate.target_iter(n_qubits):
                    pulses += self._FIXED[name](offset=3 * t)
            elif name in self._ROTATIONS:
                for t in gate.target_iter(n_qubits):
                    pulses += self._ROTATIONS[name](gate.theta, offset=3 * t)
            elif name == 'cx':
                for c, t in gate.control_target_iter(n_qubits):
                    pulses += sequences.cx_sequence(3 * c, 3 * t)
            elif name == 'cz':
                for c, t in gate.control_target_iter(n_qubits):
                    pulses += sequences.cz_sequence(3 * c, 3 * t)
            elif name == 'swap':
                for a, b in gate.control_target_iter(n_qubits):
                    pulses += sequences.swap_sequence(3 * a, 3 * b)
            else:
                raise ValueError(
                    f"Gate '{name}' is not supported by the exchange-only "
                    "transpiler. Decompose it into "
                    "x/y/z/h/s/t/rx/ry/rz/cx/cz/swap first.")
        return sequences.sequence_to_circuit(pulses, 3 * n_qubits)


register_backend('eo', EOTranspiler, overwrite=True)
