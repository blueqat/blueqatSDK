import os
import sys
import torch

# 1. 自作SDKの探索パスを設定
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from blueqat import pauli, vqe
from blueqat.backends.torch_backend import TorchBackend

def maxcut_qaoa(n_step, edges, sampler=None):
    """Setup Modern QAOA for Max-Cut problem using PyTorch VQE.

    :param n_step: QAOAのステップ数
    :param edges: グラフの辺リスト
    :returns Vqe オブジェクト
    """
    # ハミルトニアンの初期化
    hamiltonian = pauli.I() * 0

    # Max-Cut問題の定式化: 隣り合う頂点が「異なる状態」のときにエネルギーが下がるように設計
    for i, j in edges:
        hamiltonian += pauli.Z(i) * pauli.Z(j)

    hamiltonian = hamiltonian.simplify()

    # 💡 2026年最新仕様: 引数は ansatz と sampler のみでシンプルに構築
    ansatz = vqe.QaoaAnsatz(hamiltonian, n_step)
    return vqe.Vqe(ansatz, sampler=sampler)

if __name__ == "__main__":
    print("==================================================")
    print("🎨 自作 TorchBackend + 新型VQE による Max-Cut 最適化")
    print("==================================================")
    
    # グラフの定義とランナーの生成
    graph_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)]
    runner = maxcut_qaoa(2, graph_edges)
    
    # 💡 パラメータをじっくり更新させるために max_iter=300 で実行
    # 裏側では自動的に PyTorch Adam と自作の TorchBackend が駆動します
    result = runner.run(max_iter=300, verbose=True)
    
    # 最も確率の高かったビット配置（グループ分け）を取得
    best_config = result.most_common()[0][0]
    
    print("--------------------------------------------------")
    print("✨ 最適化完了！グラフのカット結果を出力します:")
    print(f"頂点配置 (q0, q1, q2, q3, q4) = {best_config}")
    
    # アスキーアートに結果の 0 / 1 をマッピングして表示
    print("""
         {4}
        / \\
       {0}---{3}
       | x |
       {1}---{2}
""".format(*best_config))
    print("==================================================")