回路とゲート
============

回路の構築
----------

:class:`~blueqat.circuit.Circuit` は操作のリストを保持します。ゲートは属性、
量子ビットは ``[...]`` で選択し、パラメータ付きゲートはインデックスの前に
呼び出しでパラメータを渡します。すべてチェーンできます:

.. code-block:: python

   import math
   from blueqat import Circuit

   Circuit().h[0].cx[0, 1].rz(math.pi / 4)[1].m[:]

量子ビット0は常に状態ベクトルインデックスの最下位ビットです（ ``'10'`` は
量子ビット1が1、量子ビット0が0。QiskitのStatevectorと同じ規約です）。

ゲートセット
------------

1量子ビットゲート
   ``i``, ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``,
   ``sxdg``, ``phase(theta)`` （別名 ``p``, ``r`` ）, ``rx(theta)``,
   ``ry(theta)``, ``rz(theta)``, ``u(theta, phi, lam[, gamma])``,
   ``mat1(matrix)`` （任意の2x2ユニタリ）。

2量子ビットゲート
   ``cx`` （別名 ``cnot`` ）, ``cy``, ``cz``, ``ch``, ``swap``, ``iswap``,
   ``iswapdg``, ``cphase(theta)`` （別名 ``cp``, ``cr`` ）, ``crx``, ``cry``,
   ``crz``, ``cu(theta, phi, lam[, gamma])``, ``rxx(theta)``, ``ryy(theta)``,
   ``rzz(theta)``, ``zz``, ``zzdg``, ``exch(theta)`` （ハイゼンベルク交換
   パルス。:doc:`exchange_only` を参照）。

3量子ビットゲート
   ``ccx`` （別名 ``toffoli`` ）, ``ccz``, ``cswap`` 。

その他の操作
   ``m`` / ``measure`` （ ``m(key="name")`` でキー付き中間測定）, ``reset``,
   ``barrier`` 。

パラメータを取らないゲートにパラメータを渡すと ``ValueError`` になります
（例: ``x(0.5)[0]`` は黙って無視されず拒否されます）。

回路の情報取得
--------------

.. code-block:: python

   c = Circuit(3).h[:].cx[0, 1].cx[1, 2].m[:]
   c.n_qubits      # 3
   c.depth()       # 4  (並列ゲートは1段と数える。barrierは数えない)
   c.count_ops()   # Counter({'h': 3, 'cx': 2, 'measure': 3})

測定確率（微分可能・指定量子ビットへの周辺化に対応）とハミルトニアン
期待値:

.. code-block:: python

   from blueqat.utils import Z

   Circuit(2).h[0].cx[0, 1].probs()          # tensor([0.5, 0., 0., 0.5])
   Circuit(2).h[0].cx[0, 1].probs([1])       # 量子ビット1の周辺確率
   Circuit(1).rx(0.4)[0].expect(1.0 * Z[0])  # <Z> = cos(0.4)

逆回路
------

:meth:`~blueqat.circuit.Circuit.dagger` はエルミート共役（ゲートを逆順に
して共役化）を返します。測定とリセットには逆操作がないため例外になります
が、 ``dagger(ignore_measurement=True)`` なら除去して続行します:

.. code-block:: python

   c = Circuit(3)  # ... 構築 ...
   identity = c + c.dagger()   # |0...0> に逆計算で戻る

OpenQASM 2.0
------------

.. code-block:: python

   qasm = Circuit(2).h[0].cx[0, 1].to_qasm()

   from blueqat.circuit_funcs import from_qasm
   c = from_qasm(qasm)

JSONシリアライズ
----------------

回路はバージョン付きのJSON互換スキーマでラウンドトリップできます
（クラウド送信のワイヤ形式でもあります）:

.. code-block:: python

   from blueqat.circuit_funcs.json_serializer import serialize, deserialize

   data = serialize(Circuit(2).h[0].cx[0, 1])
   c = deserialize(data)

回路描画
--------

``run(backend='draw')`` でmatplotlibによる回路図を描画します。登録済みの
全ゲートが描画可能で、未知の（ユーザー登録）ゲートは ``UserWarning`` 付き
で省略されます。

アンシラ量子ビット
------------------

.. code-block:: python

   c = Circuit(4).h[:]
   with c.ancilla() as a:        # 新しい量子ビットを確保
       c.cx[0, a[0]]
       c.cx[0, a[0]]
   # ブロックを出るとアンシラは |0> にリセットされます (reset=True がデフォルト)

マクロとカスタムゲート
----------------------

関数を回路メソッドとして、あるいはゲートクラスをゲートセットに登録できます:

.. code-block:: python

   from blueqat import BlueqatGlobalSetting
   from blueqat.decorators import circuitmacro

   @circuitmacro
   def bell(c, a, b):
       return c.h[a].cx[a, b]

   Circuit(2).bell(0, 1)

   BlueqatGlobalSetting.register_gate('mygate', MyGateClass)
