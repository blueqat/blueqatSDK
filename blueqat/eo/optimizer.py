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
"""Differentiable synthesis of logical EO gates as short exchange-pulse
sequences, using PyTorch autograd (the whole pipeline -- pulse areas ->
exchange matrices -> logical block -> fidelity -- is differentiable).

This is what allows going beyond the fixed analytic gate tables: any target
SU(2) can be compiled into a few constant-amplitude pulses."""

import math
from typing import List, Optional, Tuple

import torch

from .encoding import codeword_basis

Pulse = Tuple[Tuple[int, int], float]

_TWO_PI = 2.0 * math.pi


def _exchange_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable 4x4 exchange unitary (same convention as ExchangeGate)."""
    one = torch.ones((), dtype=torch.complex128)
    zero = torch.zeros((), dtype=torch.complex128)
    e = torch.exp(1j * theta.to(torch.complex128))
    a = (one + e) * 0.5
    b = (one - e) * 0.5
    return torch.stack([
        one, zero, zero, zero,
        zero, a, b, zero,
        zero, b, a, zero,
        zero, zero, zero, one
    ]).reshape(4, 4)


def _pulse_unitary_3spin(pair: Tuple[int, int], theta: torch.Tensor) -> torch.Tensor:
    """8x8 unitary of one exchange pulse on a 3-spin triple."""
    e = _exchange_matrix(theta)
    eye2 = torch.eye(2, dtype=torch.complex128)
    if pair == (0, 1):
        return torch.kron(eye2, e)
    if pair == (1, 2):
        return torch.kron(e, eye2)
    raise ValueError('pair must be (0, 1) or (1, 2).')


def synthesize_1q(target: torch.Tensor,
                  n_pulses: int = 4,
                  n_restarts: int = 8,
                  max_iter: int = 400,
                  fidelity_goal: float = 1.0 - 1e-9,
                  seed: Optional[int] = 0,
                  offset: int = 0) -> List[Pulse]:
    """Synthesize a logical 1-qubit gate as `n_pulses` exchange pulses
    alternating on pairs (0,1) and (1,2) of one triple.

    Returns the pulse sequence in application order (compatible with
    `sequences.sequence_to_circuit`). Raises RuntimeError if no restart
    reaches `fidelity_goal` -- some targets need more pulses (4 suffices
    for generic SU(2) with these two 120-degree-tilted rotation axes).
    """
    target = torch.as_tensor(target, dtype=torch.complex128)
    if target.shape != (2, 2):
        raise ValueError('target must be a 2x2 unitary.')
    basis = codeword_basis('+')
    pairs = [(0, 1) if k % 2 == 0 else (1, 2) for k in range(n_pulses)]

    if seed is not None:
        torch.manual_seed(seed)

    best: Optional[Tuple[float, torch.Tensor]] = None
    for _ in range(n_restarts):
        thetas = torch.rand(n_pulses, dtype=torch.float64) * _TWO_PI
        thetas.requires_grad_(True)
        opt = torch.optim.Adam([thetas], lr=0.1)
        for _ in range(max_iter):
            opt.zero_grad()
            u = torch.eye(8, dtype=torch.complex128)
            for pair, th in zip(pairs, thetas):
                u = _pulse_unitary_3spin(pair, th) @ u
            block = basis.conj().T @ u @ basis
            tr = torch.trace(block.conj().T @ target)
            loss = 1.0 - (tr.abs() ** 2) / 4.0
            loss.backward()
            opt.step()
            if loss.item() < 1.0 - fidelity_goal:
                break
        fid = 1.0 - loss.item()
        if best is None or fid > best[0]:
            best = (fid, thetas.detach().clone())
        if fid >= fidelity_goal:
            break

    assert best is not None
    fid, thetas = best
    if fid < fidelity_goal:
        raise RuntimeError(
            f'synthesis reached fidelity {fid:.12f} < goal {fidelity_goal}; '
            'try more pulses (n_pulses) or more restarts.')
    return [((offset + pair[0], offset + pair[1]), float(th) % _TWO_PI)
            for pair, th in zip(pairs, thetas)]
