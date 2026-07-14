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
Supports both pure Statevector and ultra-scalable Tensor Network contraction.
Leverages opt_einsum for path optimization while executing fully via PyTorch.
"""

from collections import Counter
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import opt_einsum as oe

from ..gate import *
from .backendbase import Backend

DEFAULT_SHOTS: int = 1024


def _collect_measured_qubits(gates: List[Operation], n_qubits: int) -> Optional[set]:
    """Qubit indices covered by any `measure`/`.m[...]` gate in the circuit, or None if
    the circuit has no explicit measurement at all (meaning: report every qubit, the
    long-standing default for plain `.run(shots=N)` with no `.m[...]`)."""
    measured: set = set()
    for gate in gates:
        if gate.lowername == 'measure':
            measured.update(gate.target_iter(n_qubits))
    return measured if measured else None


class TorchBackendContext:
    """Execution context holding the PyTorch quantum state or tensor network graph."""
    def __init__(self, n_qubits: int, mode: str, device: torch.device, dtype: torch.dtype,
                 initial: Optional[torch.Tensor] = None) -> None:
        self.n_qubits = n_qubits
        self.mode = "tensornet" if mode in ("tensornet", "torch_tn") else "statevector"
        self.device = device
        self.dtype = dtype
        self.cregs: List[int] = [0] * n_qubits
        self.sample: Dict[str, List[int]] = {}

        if self.mode == "statevector":
            if initial is not None:
                self.state = torch.as_tensor(initial, dtype=dtype, device=device).clone()
            else:
                self.state = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
                self.state[0] = 1.0
            self.buf = torch.zeros(1 << n_qubits, dtype=dtype, device=device)
            self.indices = torch.arange(1 << n_qubits, dtype=torch.long, device=device)
        elif self.mode == "tensornet":
            self.current_qubit_axis = list(range(n_qubits))
            self.next_axis_id = n_qubits

            if initial is not None:
                # A user-supplied initial state may be entangled across qubits, so it
                # can't be split into independent rank-1 tensors like the default |0...0>.
                # Reshape it into one dense rank-n tensor instead (bit t == qubit t, so
                # the reshape's most-significant axis is qubit n-1).
                init_t = torch.as_tensor(initial, dtype=dtype, device=device).reshape((2,) * n_qubits)
                self.tensors = [init_t]
                self.tensor_indices = [list(reversed(range(n_qubits)))]
            else:
                # 💡 メモリ爆発を防ぐため、1<<n_qubits の一括テンソルは絶対に作りません。
                self.tensors = []
                self.tensor_indices = []
                for i in range(n_qubits):
                    v = torch.zeros(2, dtype=dtype, device=device)
                    v[0] = 1.0
                    self.tensors.append(v)
                    self.tensor_indices.append([i])


class TorchBackend(Backend):
    """Unified PyTorch simulator backend supporting Autograd optimization."""

    def __init__(self, mode: str = "tensornet", device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        # 💡 デフォルトを "tensornet" に設定
        self.mode = "tensornet" if mode in ("tensornet", "torch_tn") else "statevector"
        
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.complex128

        self._init_gate_matrices()

    def copy(self) -> 'TorchBackend':
        """Return a copy of this backend. TorchBackend keeps no run-to-run cache, so
        this simply constructs a fresh instance with the same configuration."""
        return TorchBackend(mode=self.mode, device=self.device, dtype=self.dtype)

    def _init_gate_matrices(self) -> None:
        self._gate_matrices = {
            'x': lambda dev, dt: torch.tensor([[0.+0.j, 1.+0.j], [1.+0.j, 0.+0.j]], dtype=dt, device=dev),
            'y': lambda dev, dt: torch.tensor([[0.+0.j, -1.j], [1.j, 0.+0.j]], dtype=dt, device=dev),
            'z': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, -1.+0.j]], dtype=dt, device=dev),
            'h': lambda dev, dt: torch.tensor([[1.+0.j, 1.+0.j], [1.+0.j, -1.+0.j]], dtype=dt, device=dev) * (1.0 / math.sqrt(2)),
            't': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, torch.tensor(complex(1/math.sqrt(2), 1/math.sqrt(2)), dtype=dt, device=dev)]], dtype=dt, device=dev),
            's': lambda dev, dt: torch.tensor([[1.+0.j, 0.+0.j], [0.+0.j, 1.j]], dtype=dt, device=dev),
            'cx': lambda dev, dt: torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=dt, device=dev).view(2,2,2,2),
            'cz': lambda dev, dt: torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=dt, device=dev).view(2,2,2,2),
            
            # 💡 【追加】SWAPゲートの4x4行列定義を2x2x2x2テンソルとして追加
            'swap': lambda dev, dt: torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=dt, device=dev).view(2,2,2,2),
            
            # 💡 CRZゲート(動的関数)
            # 💡 torch.tensor([...]) にテンソル要素をリストで渡すと計算グラフが切断される
            #    (tensornet モードはこれがデフォルトのため autograd が壊れていた)。
            #    CRZGate.matrix() と同じく torch.diag(torch.stack([...])) で組んで勾配を維持する。
            'crz': lambda dev, dt: lambda gate: (lambda theta=torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=torch.float64, device=dev): torch.diag(torch.stack([
                torch.ones((), dtype=dt, device=dev),
                torch.ones((), dtype=dt, device=dev),
                torch.exp(-1j * theta * 0.5).to(dt),
                torch.exp(1j * theta * 0.5).to(dt),
            ])).view(2, 2, 2, 2))()
        }

    def _run_inner(self, ctx: TorchBackendContext, gates: List[Operation], n_qubits: int) -> TorchBackendContext:
        # 💡 measure/reset を挟まない高速パス専用。実行前に has_reset のない回路でのみ呼ばれる。
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
        elif name == 'swap':
            for c, t in gate.control_target_iter(ctx.n_qubits):
                nq = q.clone()
                c0, c1 = (idxs & (1 << c)) == 0, (idxs & (1 << c)) != 0
                t0, t1 = (idxs & (1 << t)) == 0, (idxs & (1 << t)) != 0
                nq[c1 & t0], nq[c0 & t1] = q[c0 & t1], q[c1 & t0]
                q, nq = nq, q
        elif isinstance(gate, OneQubitGate):
            mat = gate.matrix().to(dtype=ctx.dtype, device=ctx.device)
            for t in gate.target_iter(ctx.n_qubits):
                m = 1 << t
                t0, t1 = (idxs & m) == 0, (idxs & m) != 0
                nq[t0] = mat[0, 0] * q[t0] + mat[0, 1] * q[t1]
                nq[t1] = mat[1, 0] * q[t0] + mat[1, 1] * q[t1]
                q, nq = nq, q
        elif isinstance(gate, TwoQubitGate):
            mat = gate.matrix().to(dtype=ctx.dtype, device=ctx.device)
            for c, t in gate.control_target_iter(ctx.n_qubits):
                nq = q.clone()
                mc, mt = 1 << c, 1 << t
                masks = {
                    (bc, bt): ((idxs & mc) != 0 if bc else (idxs & mc) == 0) &
                              ((idxs & mt) != 0 if bt else (idxs & mt) == 0)
                    for bc in (0, 1) for bt in (0, 1)
                }
                # TwoQubitGate.matrix() is defined with control as the less-significant
                # bit of its 2-qubit sub-basis (row/col = target*2 + control), so index
                # accordingly rather than assuming control is the more-significant bit.
                for bc, bt in masks:
                    row = bt * 2 + bc
                    acc = 0
                    for bc2, bt2 in masks:
                        col = bt2 * 2 + bc2
                        acc = acc + mat[row, col] * q[masks[(bc2, bt2)]]
                    nq[masks[(bc, bt)]] = acc
                q, nq = nq, q
        elif isinstance(gate, IFallbackOperation):
            for sub_gate in gate.fallback(ctx.n_qubits):
                ctx.state, ctx.buf = q, nq
                ctx = self._apply_statevector_gate(ctx, sub_gate)
                q, nq = ctx.state, ctx.buf
        else:
            raise ValueError(f"Unsupported statevector gate: {name}")

        ctx.state, ctx.buf = q, nq
        return ctx

    def _apply_tensornet_gate(self, ctx: TorchBackendContext, gate: Operation) -> TorchBackendContext:
        name = gate.lowername
        
        if name in ('rx', 'ry', 'rz', 'phase'):
            float_dt = torch.float64 if ctx.dtype == torch.complex128 else torch.float32
            theta = torch.as_tensor(getattr(gate, 'theta', 0.0), dtype=float_dt, device=ctx.device)
            if name == 'rx':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5).to(ctx.dtype), (-1j*torch.sin(theta*0.5)).to(ctx.dtype)]),
                                   torch.stack([(-1j*torch.sin(theta*0.5)).to(ctx.dtype), torch.cos(theta*0.5).to(ctx.dtype)])])
            elif name == 'ry':
                mat = torch.stack([torch.stack([torch.cos(theta*0.5).to(ctx.dtype), (-torch.sin(theta*0.5)).to(ctx.dtype)]),
                                   torch.stack([torch.sin(theta*0.5).to(ctx.dtype), torch.cos(theta*0.5).to(ctx.dtype)])])
            elif name == 'rz':
                mat = torch.stack([torch.stack([torch.exp(-1j*theta*0.5).to(ctx.dtype), torch.zeros_like(theta).to(ctx.dtype)]),
                                   torch.stack([torch.zeros_like(theta).to(ctx.dtype), torch.exp(1j*theta*0.5).to(ctx.dtype)])])
            elif name == 'phase':
                mat = torch.zeros((2, 2), dtype=ctx.dtype, device=ctx.device)
                mat[0, 0] = 1.0 + 0.0j
                mat[1, 1] = torch.exp(1j * theta).to(ctx.dtype)
                
        elif name in self._gate_matrices:
            mat_or_func = self._gate_matrices[name](ctx.device, ctx.dtype)
            if callable(mat_or_func):
                mat = mat_or_func(gate)
            else:
                mat = mat_or_func
                
        elif isinstance(gate, OneQubitGate):
            mat = gate.matrix().to(dtype=ctx.dtype, device=ctx.device)
        elif isinstance(gate, TwoQubitGate):
            # TwoQubitGate.matrix() uses control as the less-significant bit
            # (row/col = target*2 + control); reshaping gives axes
            # [target_row, control_row, target_col, control_col], so permute to
            # the [control_row, target_row, control_col, target_col] order the
            # contraction below assumes.
            mat = gate.matrix().to(dtype=ctx.dtype, device=ctx.device).view(2, 2, 2, 2).permute(1, 0, 3, 2)
        elif isinstance(gate, IFallbackOperation):
            for sub_gate in gate.fallback(ctx.n_qubits):
                ctx = self._apply_tensornet_gate(ctx, sub_gate)
            return ctx
        else:
            raise ValueError(f"Unsupported TN gate: {name}")

        if len(mat.shape) == 2:
            for t in gate.target_iter(ctx.n_qubits):
                old_axis = ctx.current_qubit_axis[t]
                new_axis = ctx.next_axis_id
                ctx.next_axis_id += 1
                
                ctx.tensors.append(mat)
                ctx.tensor_indices.append([new_axis, old_axis])
                ctx.current_qubit_axis[t] = new_axis
        else:
            # 💡 cx, cz, swap, crz などの2量子ビット演算
            for c, t in gate.control_target_iter(ctx.n_qubits):
                old_c_axis = ctx.current_qubit_axis[c]
                old_t_axis = ctx.current_qubit_axis[t]
                
                new_c_axis = ctx.next_axis_id
                new_t_axis = ctx.next_axis_id + 1
                ctx.next_axis_id += 2
                
                ctx.tensors.append(mat)
                ctx.tensor_indices.append([new_c_axis, new_t_axis, old_c_axis, old_t_axis])
                ctx.current_qubit_axis[c] = new_c_axis
                ctx.current_qubit_axis[t] = new_t_axis

        return ctx

    def _collapse_statevector_qubit(self, ctx: TorchBackendContext, target: int, force_zero: bool) -> int:
        """Probabilistically collapse `target` onto |0> or |1> (a real quantum measurement),
        renormalizing the statevector. If `force_zero`, additionally flips a |1> outcome back
        to |0> (this is what `reset` is). Returns the sampled bit (before any force-zero flip).
        """
        q, idxs = ctx.state, ctx.indices
        m = 1 << target
        t0, t1 = (idxs & m) == 0, (idxs & m) != 0
        p_zero = min(max(torch.sum(torch.abs(q[t0]) ** 2).item(), 0.0), 1.0)
        bit = 0 if torch.rand(1).item() < p_zero else 1

        if bit == 0:
            norm = max(math.sqrt(p_zero), 1e-150)
            q = torch.where(t1, torch.zeros_like(q), q) / norm
        elif force_zero:
            # Collapse onto |1> then move that (now-normalized) amplitude into the |0>
            # slots, i.e. flip the qubit back to |0> as `reset` requires.
            norm = max(math.sqrt(1.0 - p_zero), 1e-150)
            flipped = q[idxs ^ m] / norm
            q = torch.where(t0, flipped, torch.zeros_like(q))
        else:
            norm = max(math.sqrt(1.0 - p_zero), 1e-150)
            q = torch.where(t0, torch.zeros_like(q), q) / norm

        ctx.state = q
        return bit

    def _collapse_tensornet_qubit(self, ctx: TorchBackendContext, target: int, device: torch.device,
                                   dtype: torch.dtype, force_zero: bool) -> int:
        """Tensor-network equivalent of `_collapse_statevector_qubit`. Computes qubit `target`'s
        marginal P(=0) by contracting the network against its own conjugate, samples an
        outcome, and attaches a (renormalized) projector as a new node for that axis -- exactly
        like applying an ordinary 1-qubit gate. `reset` additionally chains an X-flip afterward.

        Unlike a one-shot end-of-circuit sampling pass, a collapsed qubit here still gets a
        fresh open axis (so later gates can act on it again), so *every* qubit's current axis
        -- not just `target`'s -- must be shared between the ket and bra copies below and
        implicitly summed over (a proper partial trace); only genuinely historical, already
        internally-paired axes get independently relabeled for the bra copy.
        """
        axis = ctx.current_qubit_axis[target]
        shared_labels = set(ctx.current_qubit_axis)
        remap: Dict[int, int] = {}

        def _relabel(idxs: List[int]) -> List[int]:
            out = []
            for lbl in idxs:
                if lbl in shared_labels:
                    out.append(lbl)
                else:
                    if lbl not in remap:
                        remap[lbl] = -(len(remap) + 1)
                    out.append(remap[lbl])
            return out

        proj_zero = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        contract_args: List[Any] = []
        for t, idxs in zip(ctx.tensors, ctx.tensor_indices):
            contract_args += [t, idxs]
        contract_args += [proj_zero, [axis]]
        for t, idxs in zip(ctx.tensors, ctx.tensor_indices):
            contract_args += [t.conj(), _relabel(idxs)]
        contract_args += [proj_zero, [axis]]
        contract_args.append([])

        p_zero = oe.contract(*contract_args, backend="torch").real.item()
        p_zero = min(max(p_zero, 0.0), 1.0)
        bit = 0 if torch.rand(1).item() < p_zero else 1

        norm = max(math.sqrt(p_zero if bit == 0 else 1.0 - p_zero), 1e-150)
        new_axis = ctx.next_axis_id
        ctx.next_axis_id += 1
        # This projector forms a rank-2 "gate" (new_axis, old_axis) that both selects the
        # sampled branch and renormalizes it; it only touches this qubit's own edge, so the
        # rest of the network (including any now-irrelevant history) is left untouched.
        proj_mat = torch.zeros((2, 2), dtype=dtype, device=device)
        proj_mat[bit, bit] = 1.0 / norm
        ctx.tensors.append(proj_mat)
        ctx.tensor_indices.append([new_axis, axis])
        ctx.current_qubit_axis[target] = new_axis

        if force_zero and bit == 1:
            x_mat = torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]], dtype=dtype, device=device)
            flip_axis = ctx.next_axis_id
            ctx.next_axis_id += 1
            ctx.tensors.append(x_mat)
            ctx.tensor_indices.append([flip_axis, ctx.current_qubit_axis[target]])
            ctx.current_qubit_axis[target] = flip_axis

        return bit

    def _flatten_state(self, ctx: TorchBackendContext, n_qubits: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
        """Contract (tensornet mode) into, and return, the full statevector in Blueqat
        standard order (bit t == qubit t). Only valid for n_qubits <= 28."""
        if ctx.mode == "statevector":
            return ctx.state
        if n_qubits == 0:
            return torch.tensor([1.0 + 0.0j], dtype=dtype, device=device)
        contract_args: List[Any] = []
        for t, idxs in zip(ctx.tensors, ctx.tensor_indices):
            contract_args += [t, idxs]
        out_indices = [ctx.current_qubit_axis[i] for i in range(n_qubits)]
        contract_args.append(out_indices)
        current_tensor = oe.contract(*contract_args, backend="torch")
        final_permute = [out_indices.index(ctx.current_qubit_axis[i]) for i in range(n_qubits)]
        flattened_state = current_tensor.permute(tuple(final_permute)).reshape(-1)
        indices = torch.arange(len(flattened_state), device=device)
        reversed_indices = torch.zeros_like(indices)
        for i in range(n_qubits):
            bit = (indices >> i) & 1
            reversed_indices |= (bit << (n_qubits - 1 - i))
        return flattened_state[reversed_indices]

    def _run_one_shot_with_collapse(self, gates: List[Operation], n_qubits: int, mode: str,
                                     device: torch.device, dtype: torch.dtype,
                                     initial: Optional[torch.Tensor]) -> TorchBackendContext:
        """Runs the circuit once from scratch, performing a real probabilistic collapse
        at every `measure`/`reset` gate as it's encountered (a "quantum trajectory"
        simulation). This is needed whenever `reset` is used, since its effect on the rest
        of the circuit can't be captured by computing a single final statevector/tensor
        network and sampling from it afterward.
        """
        ctx = TorchBackendContext(n_qubits, mode, device, dtype, initial=initial)
        for gate in gates:
            name = gate.lowername
            if name == 'measure':
                measured = []
                for t in gate.target_iter(n_qubits):
                    if ctx.mode == "statevector":
                        bit = self._collapse_statevector_qubit(ctx, t, force_zero=False)
                    else:
                        bit = self._collapse_tensornet_qubit(ctx, t, device, dtype, force_zero=False)
                    ctx.cregs[t] = bit
                    measured.append(bit)
                if gate.key is not None:
                    if gate.key in ctx.sample:
                        if gate.duplicated == "replace":
                            ctx.sample[gate.key] = measured
                        elif gate.duplicated == "append":
                            ctx.sample[gate.key] += measured
                        else:
                            raise ValueError("Measurement key is duplicated.")
                    else:
                        ctx.sample[gate.key] = measured
            elif name == 'reset':
                for t in gate.target_iter(n_qubits):
                    if ctx.mode == "statevector":
                        self._collapse_statevector_qubit(ctx, t, force_zero=True)
                    else:
                        self._collapse_tensornet_qubit(ctx, t, device, dtype, force_zero=True)
            elif ctx.mode == "statevector":
                ctx = self._apply_statevector_gate(ctx, gate)
            else:
                ctx = self._apply_tensornet_gate(ctx, gate)
        return ctx

    def _run_with_collapse(self, gates: List[Operation], n_qubits: int, mode: str, device: torch.device,
                            dtype: torch.dtype, initial: Optional[torch.Tensor], shots: Optional[int],
                            returns: Optional[str]) -> Any:
        if shots is None and returns not in ("shots", "samples", "statevector_and_shots"):
            # No shots requested: a single trajectory's final state is enough (and, e.g.
            # for `x[:].reset[:]`, every trajectory converges on the same state anyway).
            ctx = self._run_one_shot_with_collapse(gates, n_qubits, mode, device, dtype, initial)
            return self._flatten_state(ctx, n_qubits, device, dtype)

        n_shots = shots if shots is not None else DEFAULT_SHOTS

        if returns == "samples":
            # 各ショットの `.m(key=...)` によるキー付き測定結果をそのまま返す
            return [
                self._run_one_shot_with_collapse(gates, n_qubits, mode, device, dtype, initial).sample
                for _ in range(n_shots)
            ]

        measured_qubits = _collect_measured_qubits(gates, n_qubits)
        shots_result: Counter = Counter()
        last_state: Optional[torch.Tensor] = None
        for _ in range(n_shots):
            ctx = self._run_one_shot_with_collapse(gates, n_qubits, mode, device, dtype, initial)
            if returns == "statevector_and_shots":
                # measure/reset で実際にcollapseした後の状態を、その測定結果と対応させて返す
                last_state = self._flatten_state(ctx, n_qubits, device, dtype)
            # 明示的に測定されなかった量子ビットは '0' で報告する (measured_qubits is None
            # なら .m[...] が一つもない回路なので、従来通り全量子ビットを報告する)
            cregs = ctx.cregs if measured_qubits is None else [
                b if q in measured_qubits else 0 for q, b in enumerate(ctx.cregs)
            ]
            # Blueqat標準 (qubit0が右端) に合わせて反転して結合する
            shots_result["".join(str(b) for b in reversed(cregs))] += 1

        if returns == "statevector_and_shots":
            return last_state, shots_result
        return shots_result

    def run(self, gates: List[Operation], n_qubits: int, shots: Optional[int] = None,
            returns: Optional[str] = None, **kwargs) -> Any:

        device = kwargs.get("device", self.device)
        run_mode = kwargs.get("mode", self.mode)
        if run_mode in ("tensornet", "torch_tn"):
            run_mode = "tensornet"

        target_dtype = kwargs.get("dtype", self.dtype)
        hamiltonian = kwargs.get("hamiltonian", None)
        initial = kwargs.get("initial", None)

        # 💡 reset は途中経過の状態に確率的に依存するため、最終状態ベクトルを1回だけ
        #    計算してからサンプリングする高速パスでは表現できない。`.m(key=...)` も、
        #    測定した"その時点での"値をキー別に記録する必要があるため同様。
        #    returns="statevector_and_shots" は測定でcollapseした後の状態を測定結果と
        #    対応させて返す必要があるため、同じくその場でのcollapseが要る。
        #    これらを含む回路、または returns="samples"/"statevector_and_shots" の要求は、
        #    ショットごとに最初から再実行し、measure/reset の都度その場でcollapseする。
        needs_collapse = returns in ("samples", "statevector_and_shots") or any(
            g.lowername == 'reset' or (g.lowername == 'measure' and g.key is not None) for g in gates)
        if needs_collapse:
            return self._run_with_collapse(gates, n_qubits, run_mode, device, target_dtype, initial, shots, returns)

        ctx = TorchBackendContext(n_qubits, run_mode, device, target_dtype, initial=initial)
        ctx = self._run_inner(ctx, gates, n_qubits)

        # 💡 【1つの確率振幅の要求時】
        # 💡 以前は tensornet モードのみ対応しており、statevector モードでは黙って無視され
        #    全状態ベクトルが返っていた。両モードで対応する。
        if returns == "amplitude" or "amplitude" in kwargs:
            target_bitstr = kwargs.get("amplitude", "0" * n_qubits)
            bit_list = [int(b) for b in reversed(target_bitstr)]

            if ctx.mode == "statevector":
                index = sum(bit << i for i, bit in enumerate(bit_list))
                return ctx.state[index]

            contract_args = []
            for t, idxs in zip(ctx.tensors, ctx.tensor_indices):
                contract_args.append(t)
                contract_args.append(idxs)

            for i, bit in enumerate(bit_list):
                meas_vector = torch.zeros(2, dtype=target_dtype, device=device)
                meas_vector[bit] = 1.0
                contract_args.append(meas_vector)
                contract_args.append([ctx.current_qubit_axis[i]])

            contract_args.append([])
            result_tensor = oe.contract(*contract_args, backend="torch")
            return result_tensor

        # フル状態ベクトルの展開
        if ctx.mode == "statevector":
            flattened_state = ctx.state
        elif n_qubits == 0:
            # 0量子ビットのHilbert空間は自明 (振幅1のスカラー) なので縮約は不要
            flattened_state = torch.tensor([1.0 + 0.0j], dtype=target_dtype, device=device)
        else:
            if n_qubits > 28 and shots is None:
                raise MemoryError(f"量子ビット数({n_qubits})が大きすぎるため、全状態ベクトルを展開できません。マクロな回路では returns='amplitude' または shots を指定してください。")
            
            if n_qubits <= 28:
                contract_args = []
                for t, idxs in zip(ctx.tensors, ctx.tensor_indices):
                    contract_args.append(t)
                    contract_args.append(idxs)
                    
                out_indices = [ctx.current_qubit_axis[i] for i in range(n_qubits)]
                contract_args.append(out_indices)
                
                current_tensor = oe.contract(*contract_args, backend="torch")
                final_permute = [out_indices.index(ctx.current_qubit_axis[i]) for i in range(n_qubits)]
                flattened_state = current_tensor.permute(tuple(final_permute)).reshape(-1)

        # ビットマッピングをBlueqat標準 (bit t == qubit t, statevectorモードのネイティブ順序) に一括変換
        # 💡 statevectorモードは元々この順序でネイティブに計算されているため変換不要。
        #    tensornetモードは einsum の reshape によりネイティブ順序が逆転しているため、ここで反転する。
        if ctx.mode == "statevector" or (ctx.mode == "tensornet" and n_qubits <= 28):
            if ctx.mode == "tensornet":
                indices = torch.arange(len(flattened_state), device=device)
                reversed_indices = torch.zeros_like(indices)
                for i in range(n_qubits):
                    bit = (indices >> i) & 1
                    reversed_indices |= (bit << (n_qubits - 1 - i))
                flattened_state = flattened_state[reversed_indices]

            if hamiltonian is not None:
                h_mat = hamiltonian.to_matrix(n_qubits, device=device).to(target_dtype)
                if h_mat.is_sparse:
                    hv = torch.sparse.mm(h_mat, flattened_state.unsqueeze(1)).squeeze(1)
                else:
                    hv = h_mat @ flattened_state
                return torch.vdot(flattened_state, hv).real

            if returns == "statevector" or shots is None:
                return flattened_state

        # ==================================================
        # 🧠 【ショットサンプリング】
        # ==================================================
        n_shots = shots if shots is not None else DEFAULT_SHOTS
        shots_result: Counter[str] = Counter()
        measured_qubits = _collect_measured_qubits(gates, n_qubits)
        # 明示的に .m[...] された量子ビットのみ実測値を報告し、それ以外は '0' で埋める
        # (.m[...] が一つもない場合は従来通り全量子ビットを報告する)
        keep_mask = (1 << n_qubits) - 1 if measured_qubits is None else sum(1 << q for q in measured_qubits)

        if ctx.mode == "statevector" or (ctx.mode == "tensornet" and n_qubits <= 28):
            # flattened_state は既に完全展開・標準順序に変換済みなので、そのままサンプリングできる。
            # 💡 torch.multinomial はカテゴリ数が 2^24 を超えると使えない (n_qubits >= 25 でクラッシュ
            #    する)。逆CDFサンプリング (cumsum + searchsorted) はカテゴリ数の上限がないためこちらを使う。
            with torch.no_grad():
                probs = torch.abs(flattened_state) ** 2
                cdf = torch.cumsum(probs, dim=0)
                cdf[-1] = 1.0  # 浮動小数点誤差でcdf[-1]が1未満になるのを防ぐ
                u = torch.rand(n_shots, device=probs.device, dtype=probs.dtype)
                samples = torch.searchsorted(cdf, u)
                samples &= keep_mask
            fmt = f"0{n_qubits}b"
            for idx in samples.tolist():
                shots_result[format(idx, fmt)] += 1
            return shots_result
        else:
            # 💡 【超大規模テンソルネットワーク用】 n_qubits > 28 でフル状態ベクトルを展開できない場合の
            #    逐次的な条件付きサンプリング (量子ビットを1つずつ確定していく "perfect sampling")。
            #
            #    量子ビット i の周辺確率 P(prefix, q_i=0) は、ket側のテンソルネットワークと、
            #    それを複素共役した bra側のテンソルネットワークを、まだ確定していない残りの量子ビットの
            #    軸ラベルだけ共有させて縮約することで求める (= Σ_s |amplitude(prefix, 0, s)|^2)。
            #    振幅同士を先に和ってから絶対値を取ると (旧実装のバグ)、エンタングルした状態で
            #    誤った確率になる。
            #
            #    さらに、2番目以降の量子ビットでは同時確率 P(prefix, q_i=0) を、既に確定した
            #    prefix の確率 P(prefix) で正規化して初めて正しい条件付き確率 P(q_i=0 | prefix) になる。
            with torch.no_grad():
                for shot in range(n_shots):
                    bit_string = []
                    active_tensors = list(ctx.tensors)
                    active_indices = [list(idxs) for idxs in ctx.tensor_indices]
                    prefix_prob = 1.0

                    for i in range(n_qubits):
                        # まだ確定していない量子ビット(自分自身を含む)の軸ラベルは bra/ket で共有し、
                        # それ以外(過去のゲートに由来する内部軸)は bra 側だけ独立した負のラベルへ退避する
                        shared_labels = set(ctx.current_qubit_axis[i:])
                        remap: Dict[int, int] = {}

                        def _relabel(idxs: List[int]) -> List[int]:
                            out = []
                            for lbl in idxs:
                                if lbl in shared_labels:
                                    out.append(lbl)
                                else:
                                    if lbl not in remap:
                                        remap[lbl] = -(len(remap) + 1)
                                    out.append(remap[lbl])
                            return out

                        proj_zero = torch.tensor([1.0, 0.0], dtype=target_dtype, device=device)

                        contract_args = []
                        for t, idxs in zip(active_tensors, active_indices):
                            contract_args.append(t)
                            contract_args.append(idxs)
                        contract_args.append(proj_zero)
                        contract_args.append([ctx.current_qubit_axis[i]])

                        for t, idxs in zip(active_tensors, active_indices):
                            contract_args.append(t.conj())
                            contract_args.append(_relabel(idxs))
                        contract_args.append(proj_zero)
                        contract_args.append([ctx.current_qubit_axis[i]])

                        contract_args.append([])

                        joint_zero = oe.contract(*contract_args, backend="torch").real.item()
                        joint_zero = min(max(joint_zero, 0.0), prefix_prob)
                        cond_zero = joint_zero / prefix_prob if prefix_prob > 1e-300 else 0.0
                        cond_zero = min(max(cond_zero, 0.0), 1.0)

                        chosen_bit = 0 if torch.rand(1).item() <= cond_zero else 1
                        report_bit = chosen_bit if (measured_qubits is None or i in measured_qubits) else 0
                        bit_string.append(str(report_bit))
                        prefix_prob = joint_zero if chosen_bit == 0 else max(prefix_prob - joint_zero, 0.0)

                        fixed_vec = torch.zeros(2, dtype=target_dtype, device=device)
                        fixed_vec[chosen_bit] = 1.0
                        active_tensors.append(fixed_vec)
                        active_indices.append([ctx.current_qubit_axis[i]])

                    # bit_string[i] は qubit i の測定値。Blueqat標準 (qubit0が右端) に合わせて反転して結合する
                    shots_result["".join(reversed(bit_string))] += 1

            return shots_result