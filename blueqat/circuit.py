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
        # メソッドの先頭でインポートを追加！
        from blueqat.backends import BACKENDS
        
        try:
            return self._backends[backend_name]
        except KeyError:
            backend = BACKENDS.get(backend_name)
            if backend is None:
                raise ValueError(f"Backend {backend_name} doesn't exist.")
            # インスタンス化してキャッシュ
            if isinstance(backend, type):
                self._backends[backend_name] = backend()
            else:
                # lambda関数（ファクトリ）などの場合は呼び出す
                self._backends[backend_name] = backend()
            return self._backends[backend_name]

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
            backend_name = name[9:]
            if backend_name in BACKENDS:
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
        """Make Hermitian conjugate of the circuit."""
        ops = []
        for g in reversed(self.ops):
            try:
                ops.append(g.dagger())
            except ValueError:
                if not ignore_measurement:
                    raise ValueError(
                        'Cannot make the Hermitian conjugate of this circuit because '
                        'the circuit contains measurement.')

        copied = Circuit(self.n_qubits, ops)
        return copied

    def run(self, backend: Optional[str] = None, *args, **kwargs) -> Any:
        """Run the circuit. Passes parameters to the PyTorch-based backend."""
        # メソッドの最先頭でインポートして、NameError と循環インポートの両方を完璧に防ぐ！
        from blueqat.backends import BACKENDS, DEFAULT_BACKEND_NAME
        
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)
            
        return backend.run(self.ops, self.n_qubits, *args, **kwargs)

    def statevector(self, backend: 'BackendUnion' = None, **kwargs) -> torch.Tensor:
        """Run the circuit and get a statevector as a PyTorch Tensor to keep gradients intact."""
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
        if kwargs.get('returns'):
            raise ValueError('Circuit.shots has no argument `returns`.')
        if backend is None:
            backend = self.__get_backend(DEFAULT_BACKEND_NAME)
        elif isinstance(backend, str):
            backend = self.__get_backend(backend)

        if hasattr(backend, 'shots'):
            return backend.shots(self.ops, self.n_qubits, shots=shots, **kwargs)
        return backend.run(self.ops, self.n_qubits, shots=shots, returns='shots', **kwargs)


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
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        del BACKENDS[name]

    @staticmethod
    def set_default_backend(name: str) -> None:
        """Set the default backend to be used by `Circuit`."""
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        global DEFAULT_BACKEND_NAME
        DEFAULT_BACKEND_NAME = name

    @staticmethod
    def get_default_backend_name() -> str:
        """Get the default backend name."""
        return DEFAULT_BACKEND_NAME