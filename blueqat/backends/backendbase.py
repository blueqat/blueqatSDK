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
"""Base class and plugin registration system for Blueqat backends."""

import copy
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from ..gate import IFallbackOperation, Operation

# グローバルなバックエンド登録レジストリ
_BACKEND_REGISTRY: Dict[str, Type['Backend']] = {}


class Backend(ABC):
    """Abstract base class for all Blueqat simulation and compilation backends.

    `run` has a default template-method implementation: a backend that doesn't
    override `run` directly can instead define per-gate `gate_{lowername}(self,
    gate, ctx)` hook methods (e.g. `gate_x`, `gate_cx`), plus optionally
    `_preprocess_run`/`_postprocess_run` to build/consume its own `ctx`. See
    `QasmOutputBackend` for an example. Backends like `TorchBackend` that need
    a different execution model override `run` directly instead.
    """

    def copy(self) -> 'Backend':
        """Returns a (deep) copy of this backend. Override if a shallower/cheaper
        copy is valid for a particular backend."""
        return copy.deepcopy(self)

    def _preprocess_run(self, gates: List[Operation], n_qubits: int,
                         args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Preprocess of backend run. Backend developer can override this function."""
        return gates, None

    def _postprocess_run(self, ctx: Any) -> Any:
        """Postprocess of backend run. Backend developer can override this function."""
        return None

    def _get_action(self, gate: Operation) -> Optional[Callable]:
        return getattr(self, "gate_" + gate.lowername, None)

    def _run_gates(self, gates: List[Operation], n_qubits: int, ctx: Any) -> Any:
        """Iterate gates and call the backend's `gate_{lowername}` action for each."""
        for gate in gates:
            action = self._get_action(gate)
            if action is not None:
                ctx = action(gate, ctx)
            elif isinstance(gate, IFallbackOperation):
                ctx = self._run_gates(gate.fallback(n_qubits), n_qubits, ctx)
            else:
                raise ValueError(f"Cannot run {gate.lowername} operation on this backend")
        return ctx

    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        """Execute the quantum circuit represented by a list of gates."""
        gates, ctx = self._preprocess_run(gates, n_qubits, args, kwargs)
        ctx = self._run_gates(gates, n_qubits, ctx)
        return self._postprocess_run(ctx)


def register_backend(name: str, backend_cls: Type[Backend], overwrite: bool = False) -> None:
    """Register a new backend plugin dynamically.
    
    This allows external packages (like a quimb or cuQuantum connector) 
    to register themselves into Blueqat at runtime.
    """
    global _BACKEND_REGISTRY
    if name in _BACKEND_REGISTRY and not overwrite:
        raise ValueError(f"Backend '{name}' is already registered. Set `overwrite=True` to replace it.")
    _BACKEND_REGISTRY[name] = backend_cls


def get_backend(name: str) -> Backend:
    """Retrieve an instance of the registered backend by name."""
    global _BACKEND_REGISTRY
    if name not in _BACKEND_REGISTRY:
        # 遅延インポートによる依存性の分離（デフォルトのTorch backend）
        if name in ("torch", "statevector", "tensornet"):
            from .torch_backend import TorchBackend
            return TorchBackend(mode="statevector" if name != "tensornet" else "tensornet")
        
        raise ValueError(
            f"Backend '{name}' is not registered. "
            f"If it's an external plugin (e.g., quimb, cusv), make sure to import its connector file first."
        )
    return _BACKEND_REGISTRY[name]()