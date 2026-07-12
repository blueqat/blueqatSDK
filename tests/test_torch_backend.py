import math
import pytest
import torch
from blueqat import Circuit
from blueqat.backends.torch_backend import TorchBackend

# =========================================================
# 🧪 3. ゲート演算 ＆ 自動微分（勾配）の連動検証テスト
# =========================================================

# 💡 Blueqatの標準エンディアン (index のビット t が qubit t。qubit0がLSB) に基づく状態ベクトルの絶対値。
# これは Qiskit の Statevector と同じ並び順であり、statevector/tensornet 両バックエンドで一致する。
# 基底の並び順: [|00>, |01>, |10>, |11>] (右側が q0)
GATE_TEST_CASES = [
    # 固定ゲート（パラメータなし）
    # h[0]: q0 を重ね合わせにするため、|00> と |01> に分散する -> インデックス 0 と 1
    ("h", 0, None, [1/math.sqrt(2), 1/math.sqrt(2), 0, 0]),
    # x[0]: q0 を反転 (|00> -> |01>) -> インデックス 1
    ("x", 0, None, [0, 1, 0, 0]),
    # x[1]: q1 を反転 (|00> -> |10>) -> インデックス 2
    ("x", 1, None, [0, 0, 1, 0]),
    # パラメトリックゲート（自動微分対象の角度あり）
    # ry(pi/2)[1]: q1 を重ね合わせにするため、|00> と |10> に分散する -> インデックス 0 と 2
    ("ry", 1, math.pi / 2, [1/math.sqrt(2), 0, 1/math.sqrt(2), 0]),
    # ry(pi/4)[1]: q1 をわずかに回転 -> インデックス 0 と 2
    ("ry", 1, math.pi / 4, [math.cos(math.pi/8), 0, math.sin(math.pi/8), 0]),
    # rx(pi)[0]: q0 を反転させつつ位相変化 (|00> -> |01>) -> インデックス 1 が 1.0 になる
    ("rx", 0, math.pi, [0, 1, 0, 0]),
]

@pytest.mark.parametrize("gate_name, target, param, expected_abs", GATE_TEST_CASES)
def test_statevector_gates(gate_name, target, param, expected_abs):
    """各種量子ゲートの実行結果（絶対値）と、Autogradの勾配追跡が壊れていないかを検証"""
    
    # 回路の初期化 (2量子ビット)
    c = Circuit(2)
    
    # 特殊な __getitem__ チェーンを c = ... で完全に受け取る
    if param is not None:
        theta = torch.tensor(param, dtype=torch.float64, requires_grad=True)
        c = getattr(c, gate_name)(theta)[target]
    else:
        theta = None
        c = getattr(c, gate_name)[target]
        
    # 文字列指定 "statevector" で、ハックした TorchBackend を実行
    statevector = c.run(backend="statevector")
    
    # 1. 状態ベクトルの絶対値が、数理理論値と一致するかアサーション（ハックなしの素直な比較）
    actual_abs = torch.abs(statevector).tolist()
    for act, exp in zip(actual_abs, expected_abs):
        assert abs(act - exp) < 1e-6
        
    # 2. PyTorchバックプロパゲーションを実行し、計算グラフの導通をチェック
    if theta is not None:
        loss = torch.sum(torch.abs(statevector) ** 2)
        loss.backward()
        
        assert theta.grad is not None
        assert abs(theta.grad.item()) < 1e-6

# =========================================================
# 🔄 4. 実戦想定：PyTorch Optimizer を用いた量子最適化テスト（VQE）
# =========================================================
def test_quantum_circuit_optimization_loop():
    """PyTorchのAdam最適化エンジンを使い、変分量子回路のパラメータ（角度）を学習・収束させられるか"""
    
    # 初期角度 0.0 からスタート（勾配追跡有効）
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    
    # PyTorch標準のAdamオプティマイザに量子パラメータを登録
    optimizer = torch.optim.Adam([theta], lr=0.1)
    
    # 20エポックの最適化ループを回す
    # 目標: 回路を調整して、状態 |11> の出現確率を最大化（Lossを最小化）する
    for _ in range(20):
        optimizer.zero_grad()
        
        # 変分量子回路を構築
        c = Circuit(2).h[0].ry(theta)[1].cx[0, 1]
        state = c.run(backend="statevector")
        
        # 状態ベクトルにおける |11> の検出確率を取り出す (インデックス 3)
        prob_11 = torch.abs(state[3]) ** 2
        
        # 最小化問題にするため、確率にマイナスをかけて Loss に設定
        loss = -prob_11
        
        # 逆伝播（量子回路を突き抜けて theta まで勾配を逆算）
        loss.backward()
        optimizer.step()
        
    # 最適化終了後のパラメータで最終状態を計算
    final_state = Circuit(2).h[0].ry(theta)[1].cx[0, 1].run(backend="statevector")
    final_prob_11 = torch.abs(final_state[3]) ** 2
    
    # 初期状態の確率 0.25 から、確実に最適化が進んで 0.45 以上まで跳ね上がっているか検証
    assert final_prob_11.item() > 0.45


# =========================================================
# 💻 5. 環境互換性：マルチデバイス転送テスト
# =========================================================
def test_torch_backend_device_compatibility():
    """将来的なGPU（MacのMPSやNVIDIAのCUDA）対応を見据え、TorchBackendが破綻しないかの検証"""
    
    # 実行可能な環境（MacのGPUコア（MPS）があれば優先、なければCPU）を自動判別
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # デバイス指定を組み込んだ状態ベクトルバックエンドを直接インスタンス化
    backend = TorchBackend(mode="statevector")
    
    c = Circuit(2).h[0].cx[0, 1]
    statevector = c.run(backend=backend)
    
    # バックエンドから返ってきた出力が、NumpyではなくPyTorchのTensorオブジェクトであること
    assert isinstance(statevector, torch.Tensor)
    assert statevector.dtype == torch.complex128