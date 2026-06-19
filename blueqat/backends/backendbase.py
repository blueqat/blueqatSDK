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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type
from ..gate import Operation

# グローバルなバックエンド登録レジストリ
_BACKEND_REGISTRY: Dict[str, Type['Backend']] = {}


class Backend(ABC):
    """Abstract base class for all Blueqat simulation and compilation backends."""

    @abstractmethod
    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        """Execute the quantum circuit represented by a list of gates."""
        pass


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