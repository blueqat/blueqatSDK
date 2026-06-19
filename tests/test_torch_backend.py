import torch
import pytest
from blueqat import Circuit

# 修正パターンB: json_serializer からインポートするように修正
from blueqat.circuit_funcs.json_serializer import serialize, deserialize

# もし circuit_to_unitary や flatten も circuit_funcs 直下の別のファイル（あるいは __init__.py）
# に定義されている場合はそのままインポート。不要なら削ってください。
try:
    from blueqat.circuit_funcs import circuit_to_unitary, flatten
except ImportError:
    # まだ定義されていない場合はパス（テストの障害にならないようにする緩和策）
    circuit_to_unitary, flatten = None, None


def test_torch_backend_statevector():
    """状態ベクトルモードでの動作確認テスト"""
    c = Circuit(2).h[0].cx[0, 1]
    state = c.run(backend="torch", mode="statevector")
    
    # 状態ベクトルの要素数が 4 (2^2) であることの確認
    assert state.shape == (4,)
    # 確率の合計が 1 になることの確認
    prob_sum = torch.sum(torch.abs(state) ** 2).item()
    assert pytest.approx(prob_sum) == 1.0


def test_torch_backend_tensornet():
    """テンソルネットワークモードでの振幅抽出テスト"""
    c = Circuit(2).h[0].cx[0, 1]
    # |11> の振幅は 1/sqrt(2) ≒ 0.707
    tn_amp = c.run(backend="torch", mode="tensornet", returns="amplitude", amplitude="11")
    assert pytest.approx(torch.abs(tn_amp).item()) == 0.70710678118


def test_serialization_flow():
    """シリアライズとデシリアライズの連動テスト"""
    c = Circuit(2).h[0].cx[0, 1]
    
    # シリアライズ
    data = serialize(c)
    assert data["schema"]["name"] == "blueqat-circuit"
    assert data["n_qubits"] == 2
    
    # デシリアライズして復元
    c_restored = deserialize(data)
    assert len(c_restored.ops) == 2