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
"""The 3-spin decoherence-free-subsystem (DFS) encoding of exchange-only qubits.

One logical qubit lives in the total-spin S=1/2 sector of 3 physical spins
(spin up = ``|0>``, physical qubit 3i+k is spin k of logical qubit i, qubit 0 is
the least-significant statevector bit, as everywhere in this SDK):

    ``|0_L>`` = ``|singlet(0,1)>`` ``|up(2)>``
    ``|1_L>`` = sqrt(2/3) ``|T+(0,1)>`` ``|down(2)>`` - sqrt(1/3) ``|T0(0,1)>`` ``|up(2)>``

Each logical state comes in two "gauge" copies, the total-Sz m=+1/2 sector
above and its m=-1/2 partner; exchange acts identically on both, and any
population in the fully symmetric S=3/2 quadruplet is leakage.
"""

import math
from typing import Sequence, Tuple

import torch

_SQ2 = math.sqrt(2.0)
_SQ3 = math.sqrt(3.0)
_SQ6 = math.sqrt(6.0)


def _vec(amplitudes: dict) -> torch.Tensor:
    v = torch.zeros(8, dtype=torch.complex128)
    for idx, a in amplitudes.items():
        v[idx] = a
    return v


# m = +1/2 sector. Basis-state indices are (q2 q1 q0) bit patterns.
_KET_0L_PLUS = _vec({0b010: 1 / _SQ2, 0b001: -1 / _SQ2})
_KET_1L_PLUS = _vec({0b100: math.sqrt(2 / 3), 0b010: -1 / _SQ6, 0b001: -1 / _SQ6})

# m = -1/2 sector: defined as the (normalized) total-spin lowering S_- of the
# m = +1/2 codewords, so that exchange acts with the SAME 2x2 logical block in
# both sectors (a sign flip here would conjugate the - sector's action by Z).
_KET_0L_MINUS = _vec({0b110: 1 / _SQ2, 0b101: -1 / _SQ2})
_KET_1L_MINUS = _vec({0b101: 1 / _SQ6, 0b110: 1 / _SQ6, 0b011: -math.sqrt(2 / 3)})

# Fully symmetric S=3/2 quadruplet (the leakage space).
_QUAD = torch.stack([
    _vec({0b000: 1.0}),
    _vec({0b001: 1 / _SQ3, 0b010: 1 / _SQ3, 0b100: 1 / _SQ3}),
    _vec({0b011: 1 / _SQ3, 0b101: 1 / _SQ3, 0b110: 1 / _SQ3}),
    _vec({0b111: 1.0}),
], dim=1)


def codeword_basis(m: str = '+') -> torch.Tensor:
    """(8, 2) matrix whose columns are ``|0_L>``, ``|1_L>`` of the requested gauge
    sector ('+' for total Sz = +1/2, '-' for -1/2)."""
    if m == '+':
        return torch.stack([_KET_0L_PLUS, _KET_1L_PLUS], dim=1)
    if m == '-':
        return torch.stack([_KET_0L_MINUS, _KET_1L_MINUS], dim=1)
    raise ValueError("m must be '+' or '-'")


def encode_state(logical_amplitudes: Sequence[Sequence[complex]],
                 m: str = '+') -> torch.Tensor:
    """Encode a product state of logical qubits into 3n physical spins.

    `logical_amplitudes[i]` is the (alpha, beta) pair of logical qubit i.
    Returns the 2**(3n) statevector (logical qubit 0's spins are physical
    qubits 0..2, i.e. the least-significant bits)."""
    basis = codeword_basis(m)
    state = None
    for amps in logical_amplitudes:
        a, b = complex(amps[0]), complex(amps[1])
        norm = math.sqrt(abs(a) ** 2 + abs(b) ** 2)
        if norm < 1e-12:
            raise ValueError('logical amplitudes must not be all zero.')
        triple = (a * basis[:, 0] + b * basis[:, 1]) / norm
        # Later logical qubits occupy more-significant bits.
        state = triple if state is None else torch.kron(triple, state)
    if state is None:
        raise ValueError('logical_amplitudes must not be empty.')
    return state


def leakage(state: torch.Tensor, triple: int = 0) -> float:
    """Population outside the S=1/2 subspace of the given 3-spin triple,
    i.e. the weight in its fully symmetric S=3/2 quadruplet."""
    n_qubits = (state.numel() - 1).bit_length()
    n_triples = n_qubits // 3
    if not 0 <= triple < n_triples:
        raise ValueError(f'triple must be in range(0, {n_triples}).')
    # Move the triple's three bits to the front: reshape so that the triple's
    # axes are contiguous, then contract with the quadruplet basis.
    t = state.reshape((2, ) * n_qubits)
    # axis of physical qubit q is (n_qubits - 1 - q)
    axes = [n_qubits - 1 - (3 * triple + k) for k in (2, 1, 0)]
    rest = [ax for ax in range(n_qubits) if ax not in axes]
    t = t.permute(axes + rest).reshape(8, -1)
    proj = _QUAD.conj().T.to(t.dtype) @ t
    return float((proj.abs() ** 2).sum().real)


def logical_action(unitary8: torch.Tensor, m: str = '+',
                   atol: float = 1e-9) -> torch.Tensor:
    """Extract the 2x2 logical action of a 3-spin (8x8) unitary.

    Raises ValueError if the unitary leaks out of the logical subspace of the
    requested gauge sector (the extracted block would then be non-unitary)."""
    basis = codeword_basis(m).to(unitary8.dtype)
    block = basis.conj().T @ unitary8 @ basis
    eye = torch.eye(2, dtype=block.dtype)
    if not torch.allclose(block @ block.conj().T, eye, atol=math.sqrt(atol)):
        raise ValueError('unitary leaks outside the logical subspace '
                         f'(sector m={m}).')
    return block


def logical_fidelity(actual: torch.Tensor, target: torch.Tensor) -> float:
    """Phase-insensitive gate fidelity ``|tr(A^dagger T)|^2 / d^2`` of two
    equally-sized unitaries."""
    d = actual.shape[0]
    tr = torch.trace(actual.conj().T @ target.to(actual.dtype))
    return float((tr.abs() ** 2 / d ** 2).real)


def two_qubit_codeword_basis(m1: str, m2: str) -> torch.Tensor:
    """(64, 4) basis of a 2-logical-qubit (6-spin) sector: columns are
    ``|00_L>``, ``|01_L>``, ``|10_L>``, ``|11_L>`` with gauge m1 for logical qubit 0
    (spins 0-2) and m2 for logical qubit 1 (spins 3-5)."""
    b1 = codeword_basis(m1)
    b2 = codeword_basis(m2)
    cols = []
    for j in range(2):      # logical qubit 1 (more significant)
        for i in range(2):  # logical qubit 0 (less significant)
            cols.append(torch.kron(b2[:, j], b1[:, i]))
    # Column order built above is |00>, |01>, |10>, |11> with logical qubit 0
    # as the least-significant logical bit.
    return torch.stack([cols[0], cols[1], cols[2], cols[3]], dim=1)


def two_qubit_logical_action(unitary64: torch.Tensor, m1: str = '+',
                             m2: str = '+', atol: float = 1e-9) -> torch.Tensor:
    """Extract the 4x4 logical action of a 6-spin unitary on the encoded pair."""
    basis = two_qubit_codeword_basis(m1, m2).to(unitary64.dtype)
    block = basis.conj().T @ unitary64 @ basis
    eye = torch.eye(4, dtype=block.dtype)
    if not torch.allclose(block @ block.conj().T, eye, atol=math.sqrt(atol)):
        raise ValueError('unitary leaks outside the 2-qubit logical subspace '
                         f'(sector m1={m1}, m2={m2}).')
    return block
