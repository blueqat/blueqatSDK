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

# バージョン情報
__version__ = "0.5.0"

# 1. 外部ユーザーが「blueqat.Circuit」や「blueqat.Gate」として扱えるようコアクラスを公開
from blueqat.circuit import Circuit
from blueqat.gate import Gate

# 2. バックエンド関連の絶対インポート
from blueqat.backends.backendbase import Backend, get_backend, register_backend
from blueqat.backends.torch_backend import TorchBackend
from blueqat.backends.draw_backend import DrawCircuit