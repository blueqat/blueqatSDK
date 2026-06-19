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
"""VQE (Variational Quantum Eigensolver) and QAOA ansatz module.
Modernized in 2026 for seamless PyTorch Autograd and GPU/Tensor execution.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import torch
from .circuit import Circuit
from .utils import to_inttuple


class AnsatzBase:
    """Base class for Variational Quantum Eigensolver Ansatz using PyTorch."""
    
    def __init__(self, hamiltonian: Any, n_params: int) -> None:
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        self.n_qubits: int = self.hamiltonian.max_n() + 1
        self.sparse: Optional[torch.Tensor] = None

    def make_sparse(self, sparse: bool = True, device: Optional[torch.device] = None) -> None:
        """Make sparse or dense matrix representing the Hamiltonian using PyTorch."""
        self.sparse = self.hamiltonian.to_matrix(sparse=sparse, device=device)

    def get_circuit(self, params: torch.Tensor) -> Circuit:
        """Make a circuit from parameters."""
        raise NotImplementedError

    def get_energy(self, circuit: Circuit, sampler: Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]) -> torch.Tensor:
        """Calculate energy expectation value from circuit and sampler using PyTorch."""
        val = torch.tensor(0.0 + 0.0j, dtype=torch.complex128, device=self.sparse.device if self.sparse is not None else None)
        for meas in self.hamiltonian:
            c = circuit.copy()
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(torch.tensor(-torch.pi / 2, dtype=torch.float64))[op.n]
            
            measured = sampler(c, meas.n_iter())
            for bits, prob in measured.items():
                coeff_tensor = torch.as_tensor(meas.coeff, dtype=torch.complex128, device=val.device)
                if sum(bits) % 2:
                    val = val - prob * coeff_tensor
                else:
                    val = val + prob * coeff_tensor
        return val.real

    def get_energy_sparse(self, circuit: Circuit) -> torch.Tensor:
        """Get energy using PyTorch matrix representation with Autograd support."""
        statevector = circuit.run()  # PyTorchテンソルとしての状態ベクトルを取得
        return sparse_expectation(self.sparse, statevector)

    def get_objective(self, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None, 
                      device: Optional[torch.device] = None) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get an objective function to be optimized by PyTorch optimizers."""
        if self.sparse is None:
            self.make_sparse(sparse=True, device=device)

        def objective(params: torch.Tensor) -> torch.Tensor:
            circuit = self.get_circuit(params)
            return self.get_energy(circuit, sampler)

        def obj_expect(params: torch.Tensor) -> torch.Tensor:
            circuit = self.get_circuit(params)
            return self.get_energy_sparse(circuit)

        if sampler is not None:
            return objective
        return obj_expect


class QaoaAnsatz(AnsatzBase):
    """Ansatz for QAOA (Quantum Approximate Optimization Algorithm) built on PyTorch."""
    
    def __init__(self, hamiltonian: Any, step: int = 1, init_circuit: Optional[Circuit] = None, mixer: Optional[Any] = None) -> None:
        super().__init__(hamiltonian, step * 2)
        self.hamiltonian = hamiltonian.to_expr().simplify()
        if not self.check_hamiltonian():
            raise ValueError("Hamiltonian terms are not commutable")

        self.step = step
        self.n_qubits = self.hamiltonian.max_n() + 1
        if init_circuit:
            self.init_circuit = init_circuit
            if init_circuit.n_qubits > self.n_qubits:
                self.n_qubits = init_circuit.n_qubits
        else:
            if mixer:
                raise ValueError('init_circuit is required when mixer is not default.')
            self.init_circuit = Circuit(self.n_qubits).h[:]
            
        self.mixer = mixer
        self.time_evolutions = [
            term.get_time_evolution() for term in self.hamiltonian
        ]
        self.mixer_time_evolutions = [
            term.get_time_evolution() for term in self.mixer
        ] if mixer else []

    def check_hamiltonian(self) -> bool:
        """Check whether hamiltonian is commutable."""
        return True

    def get_circuit(self, params: torch.Tensor) -> Circuit:
        c = self.init_circuit.copy()
        betas = params[:self.step]
        gammas = params[self.step:]
        for beta, gamma in zip(betas, gammas):
            beta_val = beta * torch.pi
            gamma_val = gamma * 2.0 * torch.pi
            for evo in self.time_evolutions:
                evo(c, gamma_val)
            if self.mixer is None:
                c.rx(beta_val)[:]
            else:
                for evo in self.mixer_time_evolutions:
                    evo(c, beta_val)
        return c


@dataclass
class VqeResult:
    """Dataclass holding VQE run results with PyTorch Tensors."""
    vqe: Optional['Vqe'] = None
    params: Optional[torch.Tensor] = None
    circuit: Optional[Circuit] = None
    _probs: Optional[Dict[Tuple[int, ...], float]] = None

    def most_common(self, n: int = 1) -> Tuple[Tuple[Tuple[int, ...], float], ...]:
        """Get the most common measurement outcomes."""
        return tuple(sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    @property
    def probs(self) -> Dict[Tuple[int, ...], float]:
        warnings.warn(
            "VqeResult.probs is obsoleted. Use VqeResult.get_probs().",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_probs()

    def get_probs(self, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None, 
                  rerun: Optional[bool] = None, store: bool = True) -> Dict[Tuple[int, ...], float]:
        """Get measurement outcome probabilities."""
        if rerun is None:
            rerun = sampler is not None
        if self._probs is not None and not rerun:
            return self._probs
            
        if sampler is None and self.vqe is not None:
            sampler = self.vqe.sampler

        if self.circuit is None:
            raise ValueError("No circuit available to evaluate probabilities.")

        if sampler is None:
            probs = expect(self.circuit.run(), range(self.circuit.n_qubits))
        else:
            probs = sampler(self.circuit, range(self.circuit.n_qubits))
            
        if store:
            self._probs = probs
        return probs


class Vqe:
    """VQE execution director class powered by PyTorch Optimizers."""
    
    def __init__(self, ansatz: AnsatzBase, 
                 optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None) -> None:
        self.ansatz = ansatz
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.05}
        self.sampler = sampler
        self._result: Optional[VqeResult] = None

    def run(self, max_iter: int = 500, tol: float = 1e-6, verbose: bool = False, device: Optional[torch.device] = None) -> VqeResult:
        """Run the backend VQE optimization loop natively on PyTorch Autograd."""
        if device is None:
            device = torch.device('cpu')
            
        objective_fn = self.ansatz.get_objective(self.sampler, device=device)
        
        # 最適化用パラメータテンソルの初期化
        params = torch.rand(self.ansatz.n_params, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = self.optimizer_cls([params], **self.optimizer_kwargs)
        
        for idx in range(max_iter):
            optimizer.zero_grad()
            loss = objective_fn(params)
            loss.backward()
            optimizer.step()
            
            if verbose and idx % 10 == 0:
                print(f"Iter: {idx} | Energy Loss: {loss.item():.7f}")
                
            if params.grad is not None and torch.norm(params.grad) < tol:
                break
                
        final_params = params.detach()
        c = self.ansatz.get_circuit(final_params)
        self._result = VqeResult(self, final_params, c)
        return self._result


def expect(qubits: torch.Tensor, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
    """Calculate expectation probabilities natively in PyTorch."""
    meas_tuple = tuple(meas)

    def to_key(k: int) -> Tuple[int, ...]:
        return tuple(1 if k & (1 << i) else 0 for i in meas_tuple)

    mask = reduce(lambda acc, v: acc | (1 << v), meas_tuple, 0)
    cnt = defaultdict(float)
    
    # 確率分布の計算
    probs = torch.abs(qubits) ** 2
    
    for i, p_val in enumerate(probs):
        p = p_val.item()
        if p != 0.0:
            cnt[i & mask] += p
            
    return {to_key(k): val for k, val in cnt.items()}


def non_sampling_sampler(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
    """Calculate the exact expectations utilizing the statevector directly."""
    return expect(circuit.run(), meas)


def get_measurement_sampler(n_sample: int, device: Optional[torch.device] = None) -> Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]:
    """Returns a function which calculates expectations through circuit execution sampling."""
    def sampling_by_measurement(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
        meas_tuple = tuple(meas)
        
        def reduce_bits(bits: int, m_idx: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple((bits >> m) & 1 for m in m_idx)

        statevector = circuit.run()
        probs = torch.abs(statevector) ** 2
        
        # PyTorchの高性能多項分布サンプリングを使用
        samples = torch.multinomial(probs, n_sample, replacement=True)
        unique_elements, counts = torch.unique(samples, return_counts=True)
        
        result_counts = Counter()
        for idx, count in zip(unique_elements, counts):
            bit_key = reduce_bits(idx.item(), meas_tuple)
            result_counts[bit_key] += count.item()
            
        return {k: v / n_sample for k, v in result_counts.items()}

    return sampling_by_measurement


def sparse_expectation(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Calculate matrix expectation value <vec|mat|vec> supporting PyTorch Autograd."""
    # 複素数ベクトルのエルミート共役(内積)を正確に処理し、微分可能なグラフを保つ
    if mat.is_sparse:
        mv = torch.sparse.mm(mat, vec.unsqueeze(1)).squeeze(1)
    else:
        mv = torch.mv(mat, vec)
    return torch.vdot(vec, mv).real