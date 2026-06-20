# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Backend modules registry for Blueqat.

This package manages execution engines, centering around the high-performance
PyTorch-based statevector and tensor network simulator.
"""

from typing import Any, Dict

# Jupyterなど外部からの呼び出しでも絶対に迷子にならないよう絶対インポートに統一
from blueqat.backends.backendbase import Backend, get_backend, register_backend
from blueqat.backends.torch_backend import TorchBackend
from blueqat.backends.draw_backend import DrawCircuit
from blueqat.backends.onequbitgate_transpiler import OneQubitGateCompactionTranspiler
from blueqat.backends.twoqubitgate_transpiler import TwoQubitGateDecomposingTranspiler
from .flexible_circuit_composer import FlexibleCircuitComposer

# 2026年新生Blueqatのコアバックエンドマップ
BACKENDS: Dict[str, Any] = {
    # 状態ベクトルモード (デフォルト)
    "torch": lambda: TorchBackend(mode="statevector"),
    "statevector": lambda: TorchBackend(mode="statevector"),
    
    # テンソルネットワークモード (Pure PyTorch / torch.einsum)
    "tensornet": lambda: TorchBackend(mode="tensornet"),
    
    # コンパイル & トランスパイラ
    "1q_compaction": OneQubitGateCompactionTranspiler,
    "2q_decomposition": TwoQubitGateDecomposingTranspiler,
    "composer": FlexibleCircuitComposer,
    
    # ユーティリティ
    "draw": DrawCircuit,
}

# デフォルトバックエンドを純PyTorch状態ベクトルに設定
DEFAULT_BACKEND_NAME: str = "tensornet"

# 外部からの明示的な一括インポート用リストの定義
__all__ = ['FlexibleCircuitComposer']