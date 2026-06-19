import os
import sys
import torch
import math

torch.manual_seed(42) # 乱数のシードを固定

# 1. 自作SDKの探索パスを設定
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from blueqat import Circuit, pauli
from blueqat.backends.torch_backend import TorchBackend

def run_torch_qaoa_example():
    print("==================================================")
    # 💡 2. 解きたい数値分割問題
    # ==================================================
    nums = [3, 2, 6, 9, 2, 5, 7, 3] # 合計 37 (できるだけ差が小さくなるように分ける)
    n_qubits = len(nums)
    n_step = 2  # QAOAのステップ数
    
    print(f"🧮 入力数値リスト: {nums} (要素数: {n_qubits})")
    print("🎯 目標: 合計が均等になる2つのグループに分割する")
    print("--------------------------------------------------")

    # ==================================================
    # 📐 3. ハミルトニアン（コスト関数）の設計
    # ==================================================
    hamiltonian = pauli.Expr.zero()
    for i, x in enumerate(nums):
        hamiltonian += pauli.Z[i] * x
    hamiltonian = (hamiltonian ** 2).simplify()

    # ==================================================
    # 🧠 4. PyTorchによる変分パラメータ（角度）の最適化
    # ==================================================
    angles = torch.tensor([0.1] * (2 * n_step), dtype=torch.float64, requires_grad=True)
    
    # 最適化群 Adam
    optimizer = torch.optim.Adam([angles], lr=0.1)
    
    print("🚀 PyTorch Autograd を用いた QAOA パラメータ最適化を開始します...")
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        from blueqat.vqe import QaoaAnsatz
        ansatz = QaoaAnsatz(hamiltonian, n_step)
        circuit = ansatz.get_circuit(angles)
        
        # 自作の TorchBackend で実行
        backend = TorchBackend(mode="statevector")
        statevector = circuit.run(backend=backend)
        
        probs = torch.abs(statevector) ** 2
        indices = torch.arange(len(probs), device=angles.device)
        energy_diagonal = torch.zeros_like(probs, dtype=torch.float64)
        
        # 各パウリ項の期待値を計算
        for term, coeff in hamiltonian.terms:
            # 💡 【修正ポイント】定数項（termが空）の安全な処理
            if not term:
                energy_diagonal += coeff
                continue
                
            term_sign = torch.ones_like(probs, dtype=torch.float64)
            
            # term の要素を安全に展開
            for tuple_item in term:
                # tuple_item が (qubit_idx, 'Z') であることを保証
                if isinstance(tuple_item, tuple) and len(tuple_item) == 2:
                    qubit_idx, op = tuple_item
                    if op == 'Z':
                        # ビット位置を考慮したマスク計算（左が q0 の Blueqat 仕様に準拠）
                        bit = (indices >> (n_qubits - 1 - qubit_idx)) & 1
                        sign = 1.0 - 2.0 * bit.to(torch.float64)
                        term_sign *= sign
            
            energy_diagonal += coeff * term_sign
        
        # 期待値の総和を Loss とする
        loss = torch.sum(probs * energy_diagonal)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/50 | エネルギー期待値 (Loss): {loss.item():.4f}")

    # ==================================================
    # 🏁 5. 最適化結果の解析と出力
    # ==================================================
    print("--------------------------------------------------")
    print("✨ 最適化完了！ 最終的な状態ベクトルから最頻出の組み合わせを抽出します...")
    
    final_circuit = QaoaAnsatz(hamiltonian, n_step).get_circuit(angles.detach())
    final_state = final_circuit.run(backend=TorchBackend(mode="statevector"))
    
    probs = torch.abs(final_state) ** 2
    best_idx = torch.argmax(probs).item()
    
    fmt = f"0{n_qubits}b"
    result_bits = format(best_idx, fmt)
    
    group0 = [a for a, b in zip(nums, result_bits) if b == '0']
    group1 = [a for a, b in zip(nums, result_bits) if b == '1']
    
    print(f"📊 最適なビット列: {result_bits}")
    print(f"👥 グループ 0 (合計: {sum(group0):2d}): {group0}")
    print(f"👥 グループ 1 (合計: {sum(group1):2d}): {group1}")
    print(f"⚖️ 両グループの差分: {abs(sum(group0) - sum(group1))}")
    print("==================================================")

if __name__ == "__main__":
    run_torch_qaoa_example()