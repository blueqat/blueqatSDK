微分可能な回路・VQE・QAOA
=========================

シミュレータを通した勾配
------------------------

任意のゲートパラメータに ``requires_grad=True`` の :class:`torch.Tensor`
を渡せます。ゲート行列・状態の時間発展（両実行モード）・確率・期待値まで、
パイプライン全体が微分可能なtorch演算で構成されています:

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import Z

   theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
   energy = Circuit(1).rx(theta)[0].expect(1.0 * Z[0])
   energy.backward()
   theta.grad        # -sin(0.4)、厳密な解析勾配

このため変分アルゴリズムにパラメータシフト則は不要で、通常の
``torch.optim`` のオプティマイザがそのまま使えます。

パウリ演算子とハミルトニアン
----------------------------

:mod:`blueqat.utils` がパウリ代数を提供します:

.. code-block:: python

   from blueqat.utils import X, Y, Z, I, from_qubo, qubo_bit

   h = 0.5 * Z[0] * Z[1] + 1.2 * X[0] - 3.0
   h = h.simplify()
   h.to_matrix(2)                   # 密/疎のtorch行列
   term = (X[0] * Y[1]).to_term()
   evo = term.get_time_evolution()  # exp(-i t P) を回路に追加する関数

``from_qubo`` はQUBOのコスト行列をIsingハミルトニアンに変換します。

VQE
---

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import AnsatzBase, Vqe, Z, X

   class MyAnsatz(AnsatzBase):
       def get_circuit(self, params):
           return Circuit(2).rx(params[0])[0].ry(params[1])[1].cx[0, 1]

   hamiltonian = (1.0 * Z[0] * Z[1] + 0.5 * X[0]).simplify()
   ansatz = MyAnsatz(hamiltonian, n_params=2)
   result = Vqe(ansatz).run()
   result.most_common(4)

``Vqe`` は任意の ``torch.optim`` オプティマイザクラス、オプションの
サンプラ（ショットベース推定の ``get_measurement_sampler(n)`` 、厳密で
勾配を保つ ``non_sampling_sampler`` ）、 ``initial_params`` を受け取れます。

QAOA
----

:class:`~blueqat.utils.QaoaAnsatz` は、項が互いに可換なハミルトニアン
（自動チェックされます）から標準的なQAOAアンザッツを構築します:

.. code-block:: python

   from blueqat.utils import QaoaAnsatz, Vqe, from_qubo

   qubo = [[1, 1], [1, 0]]
   h = from_qubo(qubo)
   ansatz = QaoaAnsatz(h.simplify(), step=2)
   result = Vqe(ansatz).run()
   print(result.most_common(2))

完全な自己検証付きプログラムはリポジトリの ``examples/maxcut_qaoa.py`` と
``examples/vqe_ground_state.py`` を参照してください。
