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
"""Unified Differentiable Quantum Simulator Backend (Statevector & TensorNetwork) using PyTorch."""

from collections import Counter
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch

from ..gate import *
from .backendbase import Backend

DEFAULT_SHOTS: int = 1024


class TorchBackendContext:
    """Execution context holding the PyTorch quantum state (supports both representations)."""
    def __init__(self, n_qubits: int, mode: str, device: torch.device, dtype: torch.dtype) -> None:
        self.n_qubits = n_qubits
        self.mode = mode
        self.device = device
        self.dtype = dtype
        
        if mode == "statevector":
            # 1次元の状態ベクトル: shape = (2^n,)
            self.state = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
            self.state[0] = 1.0
            self.buf = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
            self.indices = torch.arange(1 << n_qubits, dtype=torch.long, device=device)
        elif mode == "tensornet":
            # 高次元テンソルネットワーク: shape = (2, 2, ..., 2)
            self.state = torch.zeros([2] * n_qubits, dtype=dtype, device=device)
            self.state[tuple([0] * n_qubits)] = 1.0
            self.qubit_labels = [chr(97 + i) for i in range(n_qubits)]
        else:
            raise ValueError(f"Unknown mode: {mode}")


class TorchBackend(Backend):
    """Unified PyTorch simulator backend supporting Autograd, GPU, and TensorNetwork contraction."""

    def __init__(self, mode: str = "statevector", device: str = "cpu", dtype: torch.dtype = torch.complex128) -> None:
        super().__init__()
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        
        # 共通ゲートマトリックス定義 (TensorNetwork用)
        self._gate_matrices = {
            'x': lambda dev, dt: torch.tensor([[0, 1], [1, 0]], dtype=dt, device=dev),
            'y': lambda dev, dt: torch.tensor([[0, -1j], [1j, 0]], dtype=dt, device=dev),
            'z': lambda dev, dt: torch.tensor([[1, 0], [0, -1]], dtype=dt, device=dev),
            'h': lambda dev, dt: torch.tensor([[1, 1], [1, -1]], dtype=dt, device=dev) * (1.0 / math.sqrt(2)),
            't': lambda dev, dt: torch.tensor([[1, 0], [0, complex(1/math.sqrt(2), 1/math.sqrt(2))]], dtype=dt, device=dev),
            's': lambda dev, dt: torch.tensor([[1, 0], [0, 1j]], dtype=dt, device=dev),
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

    # ==========================================
    # 🪐 モード1: 状態ベクトル演算 (高速ビットマスク)
    # ==========================================
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
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=ctx.dtype, device=ctx.device)
            cos_t, sin_t = torch.cos(theta * 0.5), -1j * torch.sin(theta * 0.5)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = cos_t * q[t0] + sin_t * q[t1]
                nq[t1] = sin_t * q[t0] + cos_t * q[t1]
                q, nq = nq, q
        elif name == 'ry':
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=ctx.dtype, device=ctx.device)
            cos_t, sin_t = torch.cos(theta * 0.5), torch.sin(theta * 0.5)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = cos_t * q[t0] - sin_t * q[t1]
                nq[t1] = sin_t * q[t0] + cos_t * q[t1]
                q, nq = nq, q
        elif name == 'rz':
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=ctx.dtype, device=ctx.device)
            en, ep = torch.exp(-1j * theta * 0.5), torch.exp(1j * theta * 0.5)
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

    # ==========================================
    # 🕸️ モード2: テンソルネットワーク演算 (torch.einsum)
    # ==========================================
    def _apply_tensornet_gate(self, ctx: TorchBackendContext, gate: Operation) -> TorchBackendContext:
        name = gate.lowername
        
        # 動的マトリックス生成 (パラメトリックゲート対応)
        if name in ('rx', 'ry', 'rz'):
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=ctx.dtype, device=ctx.device)
            if name == 'rx':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5), -1j*torch.sin(theta*0.5)]),
                                   torch.stack([-1j*torch.sin(theta*0.5), torch.cos(theta*0.5)])])
            elif name == 'ry':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5), -torch.sin(theta*0.5)]),
                                   torch.stack([torch.sin(theta*0.5), torch.cos(theta*0.5)])])
            else:
                mat = torch.tensor([[torch.exp(-1j*theta*0.5), 0], [0, torch.exp(1j*theta*0.5)]], dtype=ctx.dtype, device=ctx.device)
        elif name in self._gate_matrices:
            mat = self._gate_matrices[name](ctx.device, ctx.dtype)
        elif isinstance(gate, IFallbackOperation):
            for sub_gate in gate.fallback(ctx.n_qubits):
                ctx = self._apply_tensornet_gate(ctx, sub_gate)
            return ctx
        else:
            raise ValueError(f"Unsupported TN gate: {name}")

        # einsum縮約の実行
        if len(mat.shape) == 2:  # 1-Qubit Gate
            for t in gate.target_iter(ctx.n_qubits):
                in_lbls = list(ctx.qubit_labels)
                gate_lbls = ['X', in_lbls[t]]
                out_lbls = list(in_lbls)
                out_lbls[t] = 'X'
                estr = f"{''.join(in_lbls)},{''.join(gate_lbls)}->{''.join(out_lbls)}"
                ctx.state = torch.einsum(estr, ctx.state, mat)
        else:  # 2-Qubit Gate
            for c, t in gate.control_target_iter(ctx.n_qubits):
                in_lbls = list(ctx.qubit_labels)
                gate_lbls = ['C', 'T', in_lbls[c], in_lbls[t]]
                out_lbls = list(in_lbls)
                out_lbls[c] = 'C'
                out_lbls[t] = 'T'
                estr = f"{''.join(in_lbls)},{''.join(gate_lbls)}->{''.join(out_lbls)}"
                ctx.state = torch.einsum(estr, ctx.state, mat)
        return ctx

    # ==========================================
    # 🏃 実行インターフェース
    # ==========================================
    def run(self, gates: List[Operation], n_qubits: int, shots: Optional[int] = None,
            returns: Optional[str] = None, **kwargs) -> Any:
        
        # kwargs から明示的なモード指定があれば上書き
        run_mode = kwargs.get("mode", self.mode)
        ctx = TorchBackendContext(n_qubits, run_mode, self.device, self.dtype)
        ctx = self._run_inner(ctx, gates, n_qubits)

        # テンソルネット時のピンポイント振幅抽出
        if run_mode == "tensornet" and (returns == "amplitude" or "amplitude" in kwargs):
            target_bitstr = kwargs.get("amplitude", "0" * n_qubits)
            idx = tuple(int(b) for b in reversed(target_bitstr))
            return ctx.state[idx]

        # 最終出力をフラットなBig-endian状態ベクトルに正規化
        if run_mode == "statevector":
            flattened_state = ctx.state
        else:
            flattened_state = ctx.state.permute(tuple(reversed(range(n_qubits)))).reshape(-1)

        if returns == "statevector" or shots is None:
            return flattened_state

        # サンプリング処理
        n_shots = shots if shots is not None else DEFAULT_SHOTS
        with torch.no_grad():
            probs = torch.abs(flattened_state) ** 2
            samples = torch.multinomial(probs, n_shots, replacement=True)
            
        shots_result: Counter[str] = Counter()
        fmt = f"0{n_qubits}b"
        for idx in samples.tolist():
            shots_result[format(idx, fmt)] += 1
            
        return shots_result