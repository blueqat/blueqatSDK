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
"""
This module defines Circuit and the setting for circuit.
Modernized for PyTorch Tensor Network backend integration in 2026.
"""

import warnings
from functools import partial, update_wrapper
import typing
from typing import cast, Any, Callable, Dict, Optional, Tuple, Type

import torch

from . import gate
from .gateset import get_op_type, register_operation, unregister_operation
from .typing import CircuitOperation

if typing.TYPE_CHECKING:
    from .gate import Operation
    from .backends.backendbase import Backend
    BackendUnion = typing.Union[None, str, Backend]

GLOBAL_MACROS = {}


class Circuit:
    """Store the gate operations and call the backends."""
    def __init__(self, n_qubits: int = 0, ops: Optional[list] = None):
        self.ops = ops or []
        self._backends: Dict[str, 'Backend'] = {}
        self.n_qubits = n_qubits

    def __repr__(self):
        return f'Circuit({self.n_qubits}).' + '.'.join(
            str(op) for op in self.ops)

    def __get_backend(self, backend_name):
        from blueqat.backends import BACKENDS
        from blueqat.backends.backendbase import _BACKEND_REGISTRY, get_backend

        try:
            return self._backends[backend_name]
        except KeyError:
            backend = BACKENDS.get(backend_name)
            if backend is not None:
                # インスタンス化してキャッシュ（型でもファクトリlambdaでも呼び出しは同じ）
                self._backends[backend_name] = backend()
                return self._backends[backend_name]
            if backend_name in _BACKEND_REGISTRY:
                # register_backend()経由で登録されたプラグインバックエンド
                self._backends[backend_name] = get_backend(backend_name)
                return self._backends[backend_name]
            raise ValueError(f"Backend {backend_name} doesn't exist.")

    def __backend_runner_wrapper(self, backend_name: str) -> Callable:
        backend = self.__get_backend(backend_name)

        def runner(*args, **kwargs):
            return backend.run(self.ops, self.n_qubits, *args, **kwargs)

        return runner

    def __getattr__(self, name: str) -> CircuitOperation[Any]:
        op_type = get_op_type(name)
        if op_type:
            return _GateWrapper(self, op_type)
        if name in GLOBAL_MACROS:
            macro = update_wrapper(partial(GLOBAL_MACROS[name], self), GLOBAL_MACROS[name])
            return cast(CircuitOperation[Any], macro)
        if name.startswith("run_with_"):
            # メソッド内部で遅延インポート
            from blueqat.backends import BACKENDS
            from blueqat.backends.backendbase import _BACKEND_REGISTRY
            backend_name = name[9:]
            if backend_name in BACKENDS or backend_name in _BACKEND_REGISTRY:
                return self.__backend_runner_wrapper(backend_name)
            raise AttributeError(f"Backend '{backend_name}' does not exist.")
        raise AttributeError(
            f"'Circuit' object has no attribute or gate '{name}'")

    def __add__(self, other: 'Circuit') -> 'Circuit':
        if not isinstance(other, Circuit):
            return NotImplemented
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other: 'Circuit') -> 'Circuit':
        if not isinstance(other, Circuit):
            return NotImplemented
        self.ops += other.ops
        self.n_qubits = max(self.n_qubits, other.n_qubits)
        return self

    def copy(self, copy_backends: bool = True) -> 'Circuit':
        """Copy the circuit."""
        copied = Circuit(self.n_qubits, self.ops.copy())
        if copy_backends:
            copied._backends = {k: v.copy() for k, v in self._backends.items()}
        return copied

    def dagger(self, ignore_measurement: bool = False) -> 'Circuit':
        """Make Hermitian conjugate of the circuit.

        If the circuit contains measurement or reset (which have no Hermitian
        conjugate), ValueError is raised, unless `ignore_measurement` is True,
        in which case those operations are simply dropped."""
        ops = []
        for g in reversed(self.ops):
            if not hasattr(g, 'dagger'):
                if ignore_measurement:
                    continue
                raise ValueError(
                    'Cannot make the Hermitian conjugate of this circuit because '
                    f'the circuit contains a non-invertible operation `{g.lowername}`.')
            ops.append(g.dagger())

        copied = Circuit(self.n_qubits, ops)
        return copied

    def run(self, backend: Optional[str] = None, *args, **kwargs) -> Any:
        """Run the circuit. Passes parameters to the PyTorch-based backend."""
        from blueqat.backends import BACKENDS, DEFAULT_BACKEND_NAME
        
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)
            
        return backend.run(self.ops, self.n_qubits, *args, **kwargs)

    def to_qasm(self, output_prologue: bool = True) -> str:
        """Convert this circuit into an OpenQASM 2.0 program string."""
        from blueqat.backends.qasm_output_backend import QasmOutputBackend
        return QasmOutputBackend().run(self.ops, self.n_qubits, output_prologue=output_prologue)

    def statevector(self, backend: 'BackendUnion' = None, **kwargs) -> torch.Tensor:
        """Run the circuit and get a statevector as a PyTorch Tensor to keep gradients intact."""
        from blueqat.backends import DEFAULT_BACKEND_NAME
        if kwargs.get('returns'):
            raise ValueError('Circuit.statevector has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'statevector'):
            return backend.statevector(self.ops, self.n_qubits, **kwargs)
        return backend.run(self.ops, self.n_qubits, returns='statevector', **kwargs)

    def shots(self, shots: int, backend: 'BackendUnion' = None, **kwargs) -> typing.Counter[str]:
        """Run the circuit and get shot counts as a result."""
        from blueqat.backends import DEFAULT_BACKEND_NAME
        if kwargs.get('returns'):
            raise ValueError('Circuit.shots has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'shots'):
            return backend.shots(self.ops, self.n_qubits, shots=shots, **kwargs)
        return backend.run(self.ops, self.n_qubits, shots=shots, returns='shots', **kwargs)

    def oneshot(self, backend: 'BackendUnion' = None, **kwargs) -> Tuple[torch.Tensor, str]:
        """Run the circuit once and return the post-measurement statevector together
        with the single measured bitstring."""
        from blueqat.backends import DEFAULT_BACKEND_NAME
        if kwargs.get('returns'):
            raise ValueError('Circuit.oneshot has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)
        vec, cnt = backend.run(self.ops, self.n_qubits, shots=1, returns='statevector_and_shots', **kwargs)
        return vec, next(iter(cnt))

    def _expanded_applications(self):
        """Yield (lowername, qubit-tuple) for each atomic gate application,
        expanding slices/multi-targets the same way the backends do."""
        from .gate import (Barrier, Gate, Measurement, OneQubitGate, Reset,
                           TwoQubitGate)
        n_qubits = self.n_qubits
        for op in self.ops:
            if isinstance(op, Barrier):
                yield op.lowername, tuple(op.target_iter(n_qubits))
            elif isinstance(op, (OneQubitGate, Measurement, Reset)):
                for t in op.target_iter(n_qubits):
                    yield op.lowername, (t, )
            elif isinstance(op, TwoQubitGate):
                for c, t in op.control_target_iter(n_qubits):
                    yield op.lowername, (c, t)
            elif isinstance(op, Gate):
                yield op.lowername, tuple(op.targets)
            else:
                yield op.lowername, tuple(op.target_iter(n_qubits))

    def depth(self) -> int:
        """Circuit depth: length of the longest gate sequence on any qubit path,
        counting each expanded gate application (as in Qiskit). Barriers don't
        add depth."""
        depths = [0] * self.n_qubits
        for name, qubits in self._expanded_applications():
            if name == 'barrier' or not qubits:
                continue
            d = max(depths[q] for q in qubits) + 1
            for q in qubits:
                depths[q] = d
        return max(depths, default=0)

    def count_ops(self) -> typing.Counter[str]:
        """Count expanded gate applications by name (as in Qiskit's count_ops)."""
        import collections
        return collections.Counter(name for name, _ in self._expanded_applications())

    def probs(self, qubits: Optional[typing.Sequence[int]] = None,
              backend: 'BackendUnion' = None, **kwargs) -> torch.Tensor:
        """Measurement probabilities of the circuit's final state, optionally
        marginalized onto `qubits` (as in PennyLane's `qml.probs`).

        Returns a tensor of length 2**len(qubits) where index bit j is the
        outcome of `qubits[j]` (the first listed qubit is the least-significant
        bit, matching the SDK-wide convention). Differentiable."""
        state = self.statevector(backend, **kwargs)
        p = torch.abs(state) ** 2
        if qubits is None:
            return p
        keep = list(qubits)
        if len(set(keep)) != len(keep):
            raise ValueError('qubits must not contain duplicates.')
        n = self.n_qubits
        if any(not 0 <= q < n for q in keep):
            raise ValueError(f'qubits must be in range(0, {n}).')
        # After reshape, axis k corresponds to qubit n-1-k (the statevector
        # index has qubit 0 as its least-significant bit).
        t = p.reshape((2, ) * n)
        keep_set = set(keep)
        sum_axes = [n - 1 - q for q in range(n) if q not in keep_set]
        if sum_axes:
            t = t.sum(dim=sum_axes)
        remaining = [q for q in reversed(range(n)) if q in keep_set]
        # reshape(-1) makes the first axis most significant, so order axes as
        # [last listed qubit, ..., first listed qubit].
        t = t.permute([remaining.index(q) for q in reversed(keep)])
        return t.reshape(-1)

    def expect(self, hamiltonian: Any, backend: 'BackendUnion' = None, **kwargs) -> torch.Tensor:
        """Expectation value <psi|H|psi> of a Pauli-expression Hamiltonian on
        the circuit's final state. Differentiable."""
        if hasattr(hamiltonian, 'to_expr'):
            hamiltonian = hamiltonian.to_expr().simplify()
        return self.run(backend, hamiltonian=hamiltonian, **kwargs)

    def ancilla(self, n: int = 1, pos: Optional[int] = None, stop: Optional[int] = None,
                reset: bool = True) -> '_AncillaContext':
        """Context manager allocating temporary ancilla qubit(s) for use inside the `with` block.

        By default, appends `n` fresh qubits past the circuit's current width:

            with c.ancilla() as a:
                c.cx[0, a[0]]

        `pos`/`stop` instead pin the ancilla range to specific qubit indices
        (`range(pos, stop)`; `stop` defaults to `pos + n`):

            with c.ancilla(pos=4, stop=6, reset=True) as a:
                c.cx[3, a[0]]

        If `reset` is true (the default), a `reset` gate is appended for each
        ancilla qubit on exiting the block, so they're back at ``|0>`` and safe to
        reuse elsewhere in the circuit.
        """
        if pos is not None:
            indices = list(range(pos, stop if stop is not None else pos + n))
            self.n_qubits = max(self.n_qubits, (max(indices) + 1) if indices else 0)
        else:
            indices = list(range(self.n_qubits, self.n_qubits + n))
            self.n_qubits += n
        return _AncillaContext(self, indices, reset)


class _AncillaContext:
    """Context manager returned by `Circuit.ancilla()`. See that method's docstring."""
    def __init__(self, circuit: Circuit, indices: list, reset: bool) -> None:
        self.circuit = circuit
        self.indices = indices
        self.reset = reset

    def __getitem__(self, i: int) -> int:
        return self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __enter__(self) -> '_AncillaContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.reset and exc_type is None:
            for idx in self.indices:
                self.circuit.reset[idx]


class _GateWrapper(CircuitOperation[Circuit]):
    def __init__(self, circuit: Circuit, op_type: Type['Operation']):
        self.circuit = circuit
        self.op_type = op_type
        self.params = ()
        self.options = None

    def __call__(self, *args, **kwargs) -> '_GateWrapper':
        self.params = args
        if kwargs:
            self.options = kwargs
        return self

    def __getitem__(self, targets) -> 'Circuit':
        self.circuit.ops.append(
            self.op_type.create(targets, self.params, self.options))
        self.circuit.n_qubits = max(
            gate.get_maximum_index(targets) + 1, self.circuit.n_qubits)
        return self.circuit

    def __str__(self) -> str:
        args_str = str(self.params) if self.params else ""
        if self.options:
            args_str += str(self.options)
        return self.op_type.lowername + args_str


class BlueqatGlobalSetting:
    """Setting for Blueqat."""
    @staticmethod
    def register_macro(name: str, func: Callable, allow_overwrite: bool = False) -> None:
        """Register new macro to Circuit."""
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(f"Gate '{name}' already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' already exists.")
        GLOBAL_MACROS[name] = func

    @staticmethod
    def unregister_macro(name: str) -> None:
        """Unregister a macro."""
        if name not in GLOBAL_MACROS:
            raise ValueError(f"Macro '{name}' is not registered.")
        del GLOBAL_MACROS[name]

    @staticmethod
    def register_gate(name: str, gateclass: Type['Operation'], allow_overwrite: bool = False) -> None:
        """Register new gate to gate set."""
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(f"Gate '{name}' already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' already exists.")
        register_operation(name, gateclass)

    @staticmethod
    def unregister_gate(name: str) -> None:
        """Unregister a gate from gate set."""
        if get_op_type(name) is None:
            raise ValueError(f"Gate '{name}' is not registered.")
        unregister_operation(name)

    @staticmethod
    def register_backend(name: str, backend: Type['Backend'], allow_overwrite: bool = False) -> None:
        """Register new backend."""
        from blueqat.backends import BACKENDS
        if hasattr(Circuit, "run_with_" + name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `run_with_{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `run_with_{name}`.")
        if not allow_overwrite and name in BACKENDS:
            raise ValueError(f"Backend '{name}' is already registered.")
        BACKENDS[name] = backend

    @staticmethod
    def unregister_backend(name: str) -> None:
        """Unregister a backend."""
        from blueqat.backends import BACKENDS
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        del BACKENDS[name]

    @staticmethod
    def set_default_backend(name: str) -> None:
        """Set the default backend to be used by `Circuit`."""
        from blueqat.backends import BACKENDS
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        # モジュール参照経由でグローバル変数を書き換える
        import blueqat.backends
        blueqat.backends.DEFAULT_BACKEND_NAME = name

    @staticmethod
    def get_default_backend_name() -> str:
        """Get the default backend name."""
        from blueqat.backends import DEFAULT_BACKEND_NAME
        return DEFAULT_BACKEND_NAME