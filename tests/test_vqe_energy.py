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
"""Test verification for VQE expectation and PyTorch Autograd graph integration."""

import os
import sys

# 実行環境に依存せず、ローカルの開発中リポジトリ(blueqatSDK)を最優先でPythonパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

# ⭕ 相対インポート(from .circuit ...)を完全に排除し、絶対インポートに統一
from blueqat import Circuit
from blueqat.utils import AnsatzBase, non_sampling_sampler

# ==============================================================================
# ハミルトニアンの挙動を模倣するモック（テスト用ダミー）クラス群
# ==============================================================================

class MockOp:
    def __init__(self, op_type: str, qubit_index: int):
        self.op = op_type
        self.n = qubit_index


class MockTerm:
    def __init__(self, coeff: float, ops: list):
        self.coeff = coeff
        self.ops = ops

    def n_iter(self):
        return [op.n for op in self.ops]


class MockHamiltonian:
    def __init__(self, terms: list):
        self.terms = terms

    def __iter__(self):
        return iter(self.terms)

    def max_n(self):
        max_idx = 0
        for term in self.terms:
            for op in term.ops:
                if op.n > max_idx:
                    max_idx = op.n
        return max_idx

    def to_matrix(self, sparse: bool = True, device: torch.device = None) -> torch.Tensor:
        """メジャーアップデート後のBlueqatのエンディアン仕様に合わせてダミー行列を作成。
        Qubit 0が右側、Qubit 1が左側になるため、クロネッカー積の順序を反転させます。
        """
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        I = torch.eye(2, dtype=torch.complex128)
        
        # H = 1.0 * X0 + 0.5 * Z1 
        # (Qubit 0が右なので X0 -> I ⊗ X、Qubit 1が左なので Z1 -> Z ⊗ I)
        H_mat = torch.kron(I, X) * 1.0 + torch.kron(Z, I) * 0.5
        
        if sparse:
            return H_mat.to_sparse().to(device)
        return H_mat.to(device)


# ==============================================================================
# テスト用最小限アンザッツ (Pytestのクラス収集警告を回避するためDummyプレフィックスを適用)
# ==============================================================================

class DummyAnsatz(AnsatzBase):
    """VQEエネルギー計算の微分グラフとサンプリング検証のためのテスト用最小アンザッツ"""
    def get_circuit(self, params: torch.Tensor) -> Circuit:
        # 演算内でPyTorchテンソルパラメータを使用し、計算グラフを生かす
        return Circuit(2).rx(params[0])[0].ry(params[1])[1]


# ==============================================================================
# テストケース本体
# ==============================================================================

def test_get_energy_accuracy_and_autograd():
    # 1. テスト用ハミルトニアンの生成: H = 1.0 * X0 + 0.5 * Z1
    term1 = MockTerm(1.0, [MockOp("X", 0)])
    term2 = MockTerm(0.5, [MockOp("Z", 1)])
    hamiltonian = MockHamiltonian([term1, term2])
    
    ansatz = DummyAnsatz(hamiltonian, n_params=2)
    ansatz.make_sparse(sparse=False)
    
    # 2. PyTorch Autogradを追跡させた変数テンソルを定義
    params = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    
    # 3. テストターゲット回路の生成
    circuit = ansatz.get_circuit(params)
    
    # 4. 修正した get_energy (サンプラー経由の確率集計ベース) での期待値計算
    energy_sampling = ansatz.get_energy(circuit, non_sampling_sampler)
    
    # 5. 厳密なフル行列積 (get_energy_sparse) ででの期待値計算
    energy_exact = ansatz.get_energy_sparse(circuit)
    
    # 【検証1】サンプリングによる確率期待値集計が、厳密な行列計算結果と完全に一致するか (エンディアン不一致の検出)
    assert torch.allclose(energy_sampling, energy_exact, atol=1e-6), \
        f"Mismatched calculation: Sampling={energy_sampling.item()}, Exact={energy_exact.item()}"
        
    # 6. バックプロパゲーションを走らせて微分可能性をチェック
    energy_sampling.backward()
    
    # 【検証2】PyTorchのバックエンドの計算グラフが途切れず、パラメータの grad まで勾配情報が伝播しているか
    assert params.grad is not None, "Autograd graph is broken. Gradient is None."
    assert torch.norm(params.grad) > 1e-5, "Gradient is zero. Autograd might not be tracking parameters correctly."