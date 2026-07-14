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
"""Exchange-only (EO) spin-qubit support.

Each logical qubit is encoded in 3 physical spins (a decoherence-free
subsystem), and every gate is realized purely by Heisenberg exchange pulses
(`Circuit().exch(theta)[i, j]`), the only native operation of exchange-only
silicon spin-qubit hardware.

- `encoding`: the 3-spin codewords, logical-action extraction and leakage.
- `sequences`: analytic pulse-sequence tables (RZ/X/H/... and the serial
  Fong-Wandzura CNOT).
- `transpiler`: the 'eo' backend converting a logical `Circuit` into a
  physical exchange-pulse `Circuit` (importing this package registers it).
- `optimizer`: differentiable (PyTorch autograd) synthesis of arbitrary
  logical 1-qubit gates as short pulse sequences.
"""

from . import encoding, sequences, optimizer
from .encoding import (codeword_basis, encode_state, leakage, logical_action,
                       logical_fidelity)
from .sequences import (cx_sequence, cz_sequence, h_sequence, rx_sequence,
                        ry_sequence, rz_sequence, s_sequence, sdg_sequence,
                        sequence_to_circuit, t_sequence, tdg_sequence,
                        x_sequence, y_sequence, z_sequence)
from .optimizer import synthesize_1q
from .transpiler import EOTranspiler

__all__ = [
    "codeword_basis", "encode_state", "leakage", "logical_action",
    "logical_fidelity",
    "cx_sequence", "cz_sequence", "h_sequence", "rx_sequence", "ry_sequence",
    "rz_sequence", "s_sequence", "sdg_sequence", "sequence_to_circuit",
    "t_sequence", "tdg_sequence", "x_sequence", "y_sequence", "z_sequence",
    "synthesize_1q", "EOTranspiler",
]
