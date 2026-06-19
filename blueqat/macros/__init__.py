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
"""VQE (Variational Quantum Eigensolver) and QAOA ansatz module."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
import itertools
import random
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from scipy.optimize import minimize as scipy_minimizer
from .circuit import Circuit
from .utils import to_inttuple


class AnsatzBase:
    """Base class for Variational Quantum Eigensolver Ansatz."""
    
    def __init__(self, hamiltonian: Any, n_params: int) -> None:
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        self.n_qubits: int = self.hamiltonian.max_n() + 1
        self.sparse: Optional[Any] = None

    def make_sparse(self, fmt: str = 'csc', make_method: Optional[Callable[[Any], Any]] = None) -> None:
        """Make sparse matrix representing the Hamiltonian."""
        if make_method:
            self.sparse = make_method(self.hamiltonian)
        else:
            self.sparse = self.hamiltonian.to_matrix(sparse=fmt)

    def get_circuit(self, params: np.ndarray) -> Circuit:
        """Make a circuit from parameters."""
        raise NotImplementedError

    def get_energy(self, circuit: Circuit, sampler: Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]) -> float:
        """Calculate energy expectation value from circuit and sampler."""
        val = 0.0 + 0.0j
        for meas in self.hamiltonian:
            c = circuit.copy()
            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(-np.pi / 2)[op.n]
            measured = sampler(c, meas.n_iter())
            for bits, prob in measured.items():
                if sum(bits) % 2:
                    val -= prob * meas.coeff
                else:
                    val += prob * meas.coeff
        return float(val.real)

    def get_energy_sparse(self, circuit: Circuit) -> float:
        """Get energy using a sparse matrix representation."""
        return sparse_expectation(self.sparse, circuit.run())

    def get_objective(self, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None) -> Callable[[np.ndarray], float]:
        """Get an objective function to be optimized by classical optimizer."""
        def objective(params: np.ndarray) -> float:
            circuit = self.get_circuit(params)
            circuit.make_cache()
            return self.get_energy(circuit, sampler)

        def obj_expect(params: np.ndarray) -> float:
            circuit = self.get_circuit(params)
            circuit.make_cache()
            return self.get_energy_sparse(circuit)

        if sampler is not None:
            return objective
        if self.sparse is None:
            self.make_sparse()
        return obj_expect


class QaoaAnsatz(AnsatzBase):
    """Ansatz for QAOA (Quantum Approximate Optimization Algorithm)."""
    
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
        self.init_circuit.make_cache()
        self.time_evolutions = [
            term.get_time_evolution() for term in self.hamiltonian
        ]
        self.mixer_time_evolutions = [
            term.get_time_evolution() for term in self.mixer
        ] if mixer else []

    def check_hamiltonian(self) -> bool:
        """Check whether hamiltonian is commutable."""
        return bool(self.hamiltonian.is_all_terms_commutable())

    def get_circuit(self, params: np.ndarray) -> Circuit:
        c = self.init_circuit.copy()
        betas = params[:self.step]
        gammas = params[self.step:]
        for beta, gamma in zip(betas, gammas):
            beta_val = beta * np.pi
            gamma_val = gamma * 2 * np.pi
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
    """Dataclass holding VQE run results."""
    vqe: Optional['Vqe'] = None
    params: Optional[np.ndarray] = None
    circuit: Optional[Circuit] = None
    _probs: Optional[Dict[Tuple[int, ...], float]] = None

    def most_common(self, n: int = 1) -> Tuple[Tuple[Tuple[int, ...], float], ...]:
        """Get the most common measurement outcomes."""
        return tuple(sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    @property
    def probs(self) -> Dict[Tuple[int, ...], float]:
        """Get probabilities (deprecated). Use get_probs() instead."""
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
            probs = expect(self.circuit.run(returns="statevector"), range(self.circuit.n_qubits))
        else:
            probs = sampler(self.circuit, range(self.circuit.n_qubits))
            
        if store:
            self._probs = probs
        return probs


class Vqe:
    """VQE execution director class."""
    
    def __init__(self, ansatz: AnsatzBase, minimizer: Optional[Callable[[Callable[[np.ndarray], float], int], np.ndarray]] = None, 
                 sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None) -> None:
        self.ansatz = ansatz
        self.minimizer = minimizer or get_scipy_minimizer(
            method="Powell",
            options={
                "ftol": 5.0e-2,
                "xtol": 5.0e-2,
                "maxiter": 1000
            }
        )
        self.sampler = sampler
        self._result: Optional[VqeResult] = None

    def run(self, verbose: bool = False) -> VqeResult:
        """Run the classical-quantum optimization loop."""
        objective = self.ansatz.get_objective(self.sampler)
        if verbose:
            def verbose_objective(obj: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
                def f(params: np.ndarray) -> float:
                    val = obj(params)
                    print("params:", params, "val:", val)
                    return val
                return f
            objective = verbose_objective(objective)
            
        params = self.minimizer(objective, self.ansatz.n_params)
        c = self.ansatz.get_circuit(params)
        self._result = VqeResult(self, params, c)
        return self._result

    @property
    def result(self) -> VqeResult:
        """Vqe.result is deprecated. Use `result = Vqe.run()`."""
        warnings.warn("Vqe.result is deprecated. Use `result = Vqe.run()`", DeprecationWarning, stacklevel=2)
        return self._result if self._result is not None else VqeResult()


def get_scipy_minimizer(**kwargs: Any) -> Callable[[Callable[[np.ndarray], float], int], np.ndarray]:
    """Get classical minimizer which leverages `scipy.optimize.minimize`."""
    def minimizer(objective: Callable[[np.ndarray], float], n_params: int) -> np.ndarray:
        params = np.array([random.random() for _ in range(n_params)])
        result = scipy_minimizer(objective, params, **kwargs)
        return result.x
    return minimizer


def expect(qubits: np.ndarray, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
    """Calculate perfect expectation probabilities without sampling."""
    meas_tuple = tuple(meas)

    def to_key(k: int) -> Tuple[int, ...]:
        return tuple(1 if k & (1 << i) else 0 for i in meas_tuple)

    mask = reduce(lambda acc, v: acc | (1 << v), meas_tuple, 0)
    cnt = defaultdict(float)
    
    for i, v in enumerate(qubits):
        p = float(v.real**2 + v.imag**2)
        if p != 0.0:
            cnt[i & mask] += p
            
    return {to_key(k): val for k, val in cnt.items()}


def non_sampling_sampler(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
    """Calculate the exact expectations utilizing the statevector directly."""
    return expect(circuit.run(returns="statevector"), meas)


def get_measurement_sampler(n_sample: int, run_options: Optional[Dict[str, Any]] = None) -> Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]:
    """Returns a function which calculates expectations through circuit execution sampling."""
    if run_options is None:
        run_options = {}

    def sampling_by_measurement(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
        meas_tuple = tuple(meas)
        
        def reduce_bits(bits: str, m_idx: Tuple[int, ...]) -> Tuple[int, ...]:
            bit_list = [int(x) for x in bits[::-1]]
            return tuple(bit_list[m] for m in m_idx)

        c = circuit.copy()
        c.measure[meas_tuple]
        counter = c.run(shots=n_sample, returns="shots", **run_options)
        counts = Counter({reduce_bits(bits, meas_tuple): val for bits, val in counter.items()})
        return {k: v / n_sample for k, v in counts.items()}

    return sampling_by_measurement


def get_state_vector_sampler(n_sample: int) -> Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]:
    """Returns a function which gathers expectations by sampling from a statevector distribution."""
    def sampling_by_measurement(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
        meas_tuple = tuple(meas)
        e = expect(circuit.run(returns="statevector"), meas_tuple)
        bits, probs = zip(*e.items())
        dists = np.random.multinomial(n_sample, probs) / n_sample
        return dict(zip(tuple(bits), dists))

    return sampling_by_measurement


def get_qiskit_sampler(backend: Any, **execute_kwargs: Any) -> Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]:
    """Returns a sampling function leveraging an external Qiskit backend connection."""
    try:
        import qiskit
    except ImportError:
        raise ImportError(
            "blueqat.vqe.get_qiskit_sampler() requires qiskit. Please install before calling this function."
        )
        
    shots = execute_kwargs.setdefault('shots', 1024)

    def reduce_bits(bits: str, meas_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
        if bits.startswith("0x"):
            bits_int = int(bits, base=16)
            bits = "0" * 100 + format(bits_int, "b")
        bit_list = [int(x) for x in bits[::-1]]
        return tuple(bit_list[m] for m in meas_tuple)

    def sampling(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
        meas_tuple = tuple(meas)
        if not meas_tuple:
            return {}
        c = circuit.copy()
        c.measure[meas_tuple]
        result = c.run_with_ibmq(qiskit_backend=backend, returns="qiskit_result", **execute_kwargs)
        counts = Counter({
            reduce_bits(bits, meas_tuple): val
            for bits, val in result.get_counts().items()
        })
        return {k: v / shots for k, v in counts.items()}

    return sampling


def sparse_expectation(mat: Any, vec: np.ndarray) -> float:
    """Calculate sparse matrix expectation value <vec|mat|vec>."""
    return float(np.vdot(vec, mat.dot(vec)).real)