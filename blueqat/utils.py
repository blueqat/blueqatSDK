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
"""Utilities for convenient circuit and quantum state operations.
Modernized to leverage PyTorch Autograd and native Tensor operations in 2026.
"""

from collections import Counter
from dataclasses import dataclass
import typing
from typing import Any, Dict, Iterator, Tuple, Union
import warnings

import torch

if typing.TYPE_CHECKING:
    from . import Circuit


@dataclass
class QAOAResult:
    """Result data class for QAOA optimization."""
    params: torch.Tensor
    circuit: 'Circuit'


def qaoa(hamiltonian: Any, step: int, init: typing.Optional['Circuit'] = None, 
         mixer: typing.Optional[Any] = None, max_iter: int = 200, 
         lr: float = 0.05, device: Optional[torch.device] = None) -> Union[QAOAResult, ValueError]:
    """Execute Quantum Approximate Optimization Algorithm (QAOA) using PyTorch Autograd."""
    from . import Circuit

    if device is None:
        device = torch.device('cpu')

    hamiltonian = hamiltonian.to_expr().simplify()
    N = hamiltonian.max_n()
    
    time_evolutions_cost = [
        term.get_time_evolution() for term in hamiltonian
    ] 
    
    time_evolutions_mixer = [
        term.get_time_evolution() for term in mixer
    ] if mixer else []
        
    # パラメータを PyTorch Tensor として初期化し、勾配追跡を有効化
    # 2026年現在のVQE/QAOA最適化のベストプラクティスに基づき、Adamによる自動微分を適用
    params = torch.rand(step * 2, dtype=torch.float64, device=device, requires_grad=True) * 2.0 * torch.pi
    optimizer = torch.optim.Adam([params], lr=lr)

    for i in range(max_iter):
        optimizer.zero_grad()
        
        betas = params[:step]
        gammas = params[step:]
        
        if init is None:
            c = Circuit(N).h[:]
        else:
            c = init.copy()

        for beta, gamma in zip(betas, gammas):
            for evo in time_evolutions_cost:
                evo(c, gamma)
            if mixer is None:
                c.rx(beta)[:]
            else:
                for evo in time_evolutions_mixer:
                    evo(c, beta)

        # PyTorchネイティブのテンソルシミュレータバックエンドを呼び出して期待値を計算
        loss = c.run(backend="torch_tn", hamiltonian=hamiltonian, device=device)
        
        # バックプロパゲーションの実行
        loss.backward()
        optimizer.step()
        
        # 収束チェック（非常に平坦になったら早期終了）
        if params.grad is not None and torch.norm(params.grad) < 1e-5:
            break

    # 最適化されたパラメータで最終的な回路を構築
    with torch.no_grad():
        betas = params[:step]
        gammas = params[step:]
        if init is None:
            final_circ = Circuit(N).h[:]
        else:
            final_circ = init.copy()

        for beta, gamma in zip(betas, gammas):
            for evo in time_evolutions_cost:
                evo(final_circ, gamma)
            if mixer is None:
                final_circ.rx(beta)[:]
            else:
                for evo in time_evolutions_mixer:
                    evo(final_circ, beta)

    return QAOAResult(params=params.detach(), circuit=final_circ)


def to_inttuple(
    bitstr: Union[str, Counter, Dict[str, int]]
) -> Union[Tuple[int, ...], Counter, Dict[Tuple[int, ...], int]]:
    """Convert from bit string like '01011' to int tuple like (0, 1, 0, 1, 1)."""
    if isinstance(bitstr, str):
        return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter):
        return Counter({tuple(int(b) for b in k): v for k, v in bitstr.items()})
    if isinstance(bitstr, dict):
        return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")


def ignore_global_phase(statevec: torch.Tensor) -> torch.Tensor:
    """Multiply e^-iθ to `statevec` where θ is a phase of first non-zero element using PyTorch."""
    # 勾配グラフを壊さないようにインプレース演算を避けてフェーズ調整
    mask = torch.abs(statevec) > 1e-7
    indices = torch.nonzero(mask)
    if len(indices) > 0:
        first_idx = indices[0][0]
        q = statevec[first_idx]
        ang = torch.abs(q) / q
        return statevec * ang
    return statevec


def check_unitarity(mat: torch.Tensor) -> bool:
    """Check whether mat is a unitary matrix using PyTorch linalg."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    dim = mat.shape[0]
    identity = torch.eye(dim, dtype=mat.dtype, device=mat.device)
    return torch.allclose(mat @ mat.resolve_conj().T, identity, atol=1e-7)


def circuit_to_unitary(circ: 'Circuit', *runargs: Any, **runkwargs: Any) -> torch.Tensor:
    """Convert circuit to unitary matrix."""
    warnings.warn(
        "blueqat.util.circuit_to_unitary is moved to "
        "blueqat.circuit_funcs.circuit_to_unitary.circuit_to_unitary.",
        DeprecationWarning,
        stacklevel=2
    )
    from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary as f
    return f(circ, *runargs, **runkwargs)


def calc_u_params(mat: torch.Tensor) -> Tuple[float, float, float, float]:
    """Calculate U-gate parameters from a 2x2 unitary matrix using PyTorch."""
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    
    gamma = torch.angle(mat[0, 0]).item()
    mat = mat * torch.exp(torch.tensor(-1j * gamma, dtype=torch.complex128, device=mat.device))
    
    theta = torch.atan2(torch.abs(mat[1, 0]), mat[0, 0].real).item() * 2.0
    phi_plus_lambda = torch.angle(mat[1, 1]).item()
    phi = torch.angle(mat[1, 0]).item() % (2.0 * torch.pi)
    lam = (phi_plus_lambda - phi) % (2.0 * torch.pi)
    
    return theta, phi, lam, gamma


def sqrt_2x2_matrix(mat: torch.Tensor) -> torch.Tensor:
    """Returns square root of a 2x2 matrix natively in PyTorch."""
    assert mat.shape == (2, 2)
    s = torch.sqrt(torch.linalg.det(mat))
    t = torch.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    if torch.abs(t) < 1e-8:
        s = -s
        t = torch.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    identity = torch.eye(2, dtype=mat.dtype, device=mat.device)
    return (mat + s * identity) / t


def gen_graycode(n: int) -> Iterator[int]:
    """Generate an iterator which returns Gray code."""
    return (v ^ (v >> 1) for v in range(2**n))


def gen_gray_controls(n: int) -> Iterator[Tuple[int, int, int]]:
    """Generate an iterator which returns bit indices for constructing
    Gray code based controlled gate.
    """
    def gen_changedbit(n_bits: int) -> Iterator[int]:
        pow2 = [2 ** i for i in range(n_bits)]
        gen = gen_graycode(n_bits)
        try:
            prev = next(gen)
        except StopIteration:
            raise ValueError("Empty Gray code generation.") from None
        for g in gen:
            yield pow2.index(g ^ prev)
            prev = g

    def gen_cxtarget() -> Iterator[int]:
        k = 0
        while True:
            for _ in range(2**k):
                yield k
            k += 1

    def gen_parity() -> Iterator[int]:
        while True:
            yield 0
            yield 1

    for c0, c1, p in zip(gen_changedbit(n), gen_cxtarget(), gen_parity()):
        if c0 == c1:
            yield c0 - 1, c1, p
        else:
            yield c0, c1, p