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
"""Blueqat Quantum Computing SDK core module."""

# _version.py からバージョン情報を引っ張ってくる
from blueqat._version import __version__

# 1. コアクラスとグローバル設定を公開
# (BlueqatGlobalSetting を circuit からインポートして追加します)
from blueqat.circuit import Circuit, BlueqatGlobalSetting
from blueqat.gate import Gate

# 2. バックエンド関連の絶対インポート
from blueqat.backends.backendbase import Backend, get_backend, register_backend
from blueqat.backends.torch_backend import TorchBackend
from blueqat.backends.draw_backend import DrawCircuit

# 公開するシンボルを明示的に指定（テスト環境の検出をより確実にします）
__all__ = [
    "__version__",
    "Circuit",
    "BlueqatGlobalSetting",
    "Gate",
    "Backend",
    "get_backend",
    "register_backend",
    "TorchBackend",
    "DrawCircuit",
]