はじめに
========

インストール
------------

blueqat は Python 3.11 以上が必要で、シミュレーション基盤として PyTorch を
インストールします:

.. code-block:: console

   pip install git+https://github.com/blueqat/blueqatSDK

開発用:

.. code-block:: console

   git clone https://github.com/blueqat/blueqatSDK
   cd blueqatSDK
   pip install -e .[dev]
   pytest tests/ -q

最初の回路
----------

回路はメソッドチェーンで構築します。ゲートは属性として選び、 ``[...]`` の
インデックスで量子ビットに適用します:

.. code-block:: python

   from blueqat import Circuit

   c = Circuit()            # 量子ビット数は自動で拡張されます
   c.h[0]                   # 量子ビット0にHadamard
   c.cx[0, 1]               # CNOT: 制御0、標的1

   # あるいは1つのチェーンとして:
   c = Circuit().h[0].cx[0, 1]

実行すると状態ベクトルが :class:`torch.Tensor` として返ります
（量子ビット0が状態インデックスの最下位ビットです）:

.. code-block:: python

   c.run()
   # tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j])

測定結果をサンプリングする場合:

.. code-block:: python

   c.m[:].run(shots=1000)
   # Counter({'00': 493, '11': 507})

スライスで複数の量子ビットにまとめて適用できます:

.. code-block:: python

   Circuit(4).h[:]          # 全量子ビットにH
   Circuit(4).x[1:3]        # 量子ビット1, 2にX
   Circuit(4).z[0, 3]       # 量子ビット0と3にZ

次に読むページ
--------------

- :doc:`guide/circuits` -- 全ゲートセット、回路の情報取得、QASM。
- :doc:`guide/backends` -- statevector と tensornet、shots、大規模回路。
- :doc:`guide/autograd` -- 微分可能な回路、VQEとQAOA。
- :doc:`guide/exchange_only` -- Exchange-Onlyスピン量子ビットとパルス
  コンパイル。
