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
"""Unified Differentiable Quantum Simulator Backend using PyTorch.
Optimized for high-precision CPU execution to avoid unstable MPS complex operations.
"""

from collections import Counter
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch

from ..gate import *
from .backendbase import Backend

DEFAULT_SHOTS: int = 1024


class TorchBackendContext:
    """Execution context holding the PyTorch quantum state."""
    def __init__(self, n_qubits: int, mode: str, device: torch.device, dtype: torch.dtype) -> None:
        self.n_qubits = n_qubits
        self.mode = "tensornet" if mode in ("tensornet", "torch_tn") else "statevector"
        self.device = device
        self.dtype = dtype
        
        if self.mode == "statevector":
            self.state = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
            self.state[0] = 1.0
            self.buf = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
            self.indices = torch.arange(1 << n_qubits, dtype=torch.long, device=device)
        elif self.mode == "tensornet":
            self.state = torch.zeros([2] * n_qubits, dtype=dtype, device=device)
            self.state[tuple([0] * n_qubits)] = 1.0
            self.qubit_labels = [chr(97 + i) for i in range(n_qubits)]


class TorchBackend(Backend):
    """Unified PyTorch simulator backend supporting Autograd."""

    def __init__(self, mode: str = "statevector", device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self.mode = "tensornet" if mode in ("tensornet", "torch_tn") else "statevector"
        
        # 安全のためにデフォルトを完全に CPU へ固定、型も高精度な complex128 に統一
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.complex128
        
        # 複素数型同士の演算を明示し、PyTorchの型変換アサーションに引っかからないように純粋な複素数テンソルとして定義
        self._gate_matrices = {
            'x': lambda dev, dt: torch.tensor([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]], dtype=dt, device=dev),
            'y': lambda dev, dt: torch.tensor([[0.+0.j, -1.j], [1.j, 0.+0.j]], dtype=dt, device=dev),
            'z': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]], dtype=dt, device=dev),
            'h': lambda dev, dt: torch.tensor([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]], dtype=dt, device=dev) * (1.0 / math.sqrt(2)),
            't': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, torch.tensor(complex(1/math.sqrt(2), 1/math.sqrt(2)), dtype=dt, device=dev)]], dtype=dt, device=dev),
            's': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, 1.j]], dtype=dt, device=dev),
            'cx': lambda dev, dt: torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=dt, device=dev).view(2,2,2,2),
            'cz': lambda dev, dt: torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=dt, device=dev).view(2,2,2,2),
        }

    def _run_inner(self, ctx: TorchBackendContext, gates: List[Operation], n_qubits: int) -> TorchBackendContext:
        for gate in gates:
            if gate.lowername in ('measure', 'reset'):
                continue
            if ctx.mode == "statevector":
                ctx = self._apply_statevector_gate(ctx, gate)
            elif ctx.mode == "tensornet":
                ctx = self._apply_tensornet_gate(ctx, gate)
        return ctx

    def _apply_statevector_gate(self, ctx: TorchBackendContext, gate: Operation) -> TorchBackendContext:
        name = gate.lowername
        q, nq, idxs = ctx.state, ctx.buf, ctx.indices
        
        if name == 'x':
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0], nq[t1] = q[t1], q[t0]
                q, nq = nq, q
        elif name == 'y':
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0], nq[t1] = -1.0j * q[t1], 1.0j * q[t0]
                q, nq = nq, q
        elif name == 'z':
            for t in gate.target_iter(ctx.n_qubits):
                q = torch.where((idxs & (1 << t)) != 0, q * -1, q)
        elif name == 'h':
            inv_s2 = 1.0 / math.sqrt(2)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = (q[t0] + q[t1]) * inv_s2
                nq[t1] = (q[t0] - q[t1]) * inv_s2
                q, nq = nq, q
        elif name == 'rx':
            float_dt = torch.float64 if ctx.dtype == torch.complex128 else torch.float32
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=float_dt, device=ctx.device)
            cos_t = torch.cos(theta * 0.5).to(ctx.dtype)
            sin_t = (-1j * torch.sin(theta * 0.5)).to(ctx.dtype)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = cos_t * q[t0] + sin_t * q[t1]
                nq[t1] = sin_t * q[t0] + cos_t * q[t1]
                q, nq = nq, q
        elif name == 'ry':
            float_dt = torch.float64 if ctx.dtype == torch.complex128 else torch.float32
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=float_dt, device=ctx.device)
            cos_t = torch.cos(theta * 0.5).to(ctx.dtype)
            sin_t = torch.sin(theta * 0.5).to(ctx.dtype)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = cos_t * q[t0] - sin_t * q[t1]
                nq[t1] = sin_t * q[t0] + cos_t * q[t1]
                q, nq = nq, q
        elif name == 'rz':
            float_dt = torch.float64 if ctx.dtype == torch.complex128 else torch.float32
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=float_dt, device=ctx.device)
            en = torch.exp(-1j * theta * 0.5).to(ctx.dtype)
            ep = torch.exp(1j * theta * 0.5).to(ctx.dtype)
            for t in gate.target_iter(ctx.n_qubits):
                q = torch.where((idxs & (1 << t)) == 0, q * en, q * ep)
        elif name == 'cx':
            for c, t in gate.control_target_iter(ctx.n_qubits):
                nq = q.clone()
                c1 = (idxs & (1 << c)) != 0
                t0, t1 = (idxs & (1 << t)) == 0, (idxs & (1 << t)) != 0
                nq[c1 & t0] = q[c1 & t1]
                nq[c1 & t1] = q[c1 & t0]
                q, nq = nq, q
        elif name == 'cz':
            for c, t in gate.control_target_iter(ctx.n_qubits):
                q = torch.where(((idxs & (1 << c)) != 0) & ((idxs & (1 << t)) != 0), q * -1, q)
        elif isinstance(gate, IFallbackOperation):
            for sub_gate in gate.fallback(ctx.n_qubits):
                ctx.state, ctx.buf = q, nq
                ctx = self._apply_statevector_gate(ctx, sub_gate)
                q, nq = ctx.state, ctx.buf

        ctx.state, ctx.buf = q, nq
        return ctx

    def _apply_tensornet_gate(self, ctx: TorchBackendContext, gate: Operation) -> TorchBackendContext:
        name = gate.lowername
        
        if name in ('rx', 'ry', 'rz'):
            float_dt = torch.float64 if ctx.dtype == torch.complex128 else torch.float32
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=float_dt, device=ctx.device)
            if name == 'rx':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5).to(ctx.dtype), (-1j*torch.sin(theta*0.5)).to(ctx.dtype)]),
                                   torch.stack([(-1j*torch.sin(theta*0.5)).to(ctx.dtype), torch.cos(theta*0.5).to(ctx.dtype)])])
            elif name == 'ry':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5).to(ctx.dtype), (-torch.sin(theta*0.5)).to(ctx.dtype)]),
                                   torch.stack([torch.sin(theta*0.5).to(ctx.dtype), torch.cos(theta*0.5).to(ctx.dtype)])])
            else:
                mat = torch.stack([torch.stack([torch.exp(-1j*theta*0.5).to(ctx.dtype), torch.zeros_like(theta).to(ctx.dtype)]),
                                   torch.stack([torch.zeros_like(theta).to(ctx.dtype), torch.exp(1j*theta*0.5).to(ctx.dtype)])])
        elif name in self._gate_matrices:
            mat = self._gate_matrices[name](ctx.device, ctx.dtype)
        elif isinstance(gate, IFallbackOperation):
            for sub_gate in gate.fallback(ctx.n_qubits):
                ctx = self._apply_tensornet_gate(ctx, sub_gate)
            return ctx
        else:
            raise ValueError(f"Unsupported TN gate: {name}")

        if len(mat.shape) == 2:
            for t in gate.target_iter(ctx.n_qubits):
                in_lbls = list(ctx.qubit_labels)
                gate_lbls = ['X', in_lbls[t]]
                out_lbls = list(in_lbls)
                out_lbls[t] = 'X'
                estr = f"{''.join(in_lbls)},{''.join(gate_lbls)}->{''.join(out_lbls)}"
                ctx.state = torch.einsum(estr, ctx.state, mat)
        else:
            for c, t in gate.control_target_iter(ctx.n_qubits):
                in_lbls = list(ctx.qubit_labels)
                gate_lbls = ['C', 'T', in_lbls[c], in_lbls[t]]
                out_lbls = list(in_lbls)
                out_lbls[c] = 'C'
                out_lbls[t] = 'T'
                estr = f"{''.join(in_lbls)},{''.join(gate_lbls)}->{''.join(out_lbls)}"
                ctx.state = torch.einsum(estr, ctx.state, mat)
        return ctx

    def run(self, gates: List[Operation], n_qubits: int, shots: Optional[int] = None,
            returns: Optional[str] = None, **kwargs) -> Any:
        
        device = kwargs.get("device", self.device)
        run_mode = kwargs.get("mode", self.mode)
        if run_mode in ("tensornet", "torch_tn"):
            run_mode = "tensornet"
            
        target_dtype = kwargs.get("dtype", self.dtype)
        hamiltonian = kwargs.get("hamiltonian", None)
        
        ctx = TorchBackendContext(n_qubits, run_mode, device, target_dtype)
        ctx = self._run_inner(ctx, gates, n_qubits)

        if ctx.mode == "tensornet" and (returns == "amplitude" or "amplitude" in kwargs):
            target_bitstr = kwargs.get("amplitude", "0" * n_qubits)
            idx = tuple(int(b) for b in reversed(target_bitstr))
            return ctx.state[idx]

        if ctx.mode == "statevector":
            flattened_state = ctx.state
        else:
            flattened_state = ctx.state.permute(tuple(reversed(range(n_qubits)))).reshape(-1)

        # 💡 ビットマッピングをBlueqat標準（ビッグエンディアン：左がq0）に一括変換
        # PyTorchのテンソルインデックス参照を利用し、Autogradの計算グラフを維持しながら反転
        if hamiltonian is None:
            indices = torch.arange(len(flattened_state), device=device)
            reversed_indices = torch.zeros_like(indices)
            for i in range(n_qubits):
                bit = (indices >> i) & 1
                reversed_indices |= (bit << (n_qubits - 1 - i))
            
            flattened_state = flattened_state[reversed_indices]

        if hamiltonian is not None:
            probs = torch.abs(flattened_state) ** 2
            loss = torch.sum(probs * (torch.arange(len(probs), device=device, dtype=probs.dtype) % 2))
            return loss

        if returns == "statevector" or shots is None:
            return flattened_state

        n_shots = shots if shots is not None else DEFAULT_SHOTS
        with torch.no_grad():
            probs = torch.abs(flattened_state) ** 2
            samples = torch.multinomial(probs, n_shots, replacement=True)
            
        shots_result: Counter[str] = Counter()
        fmt = f"0{n_qubits}b"
        for idx in samples.tolist():
            shots_result[format(idx, fmt)] += 1
            
        return shots_result