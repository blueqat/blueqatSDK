Exchange-Onlyスピン量子ビット
=============================

:mod:`blueqat.eo` は **Exchange-Only（EO）量子ビット** — 半導体（シリコン
量子ドット）スピンハードウェアの動作方式 — をサポートします。各論理量子
ビットは **3つの物理スピン** に符号化され、唯一のネイティブ操作は2スピン間
のハイゼンベルク交換パルスです。

交換パルス
----------

.. math::

   U(\theta) = e^{-i \frac{\theta}{2}(\mathrm{SWAP} - I)}

は三重項（対称）部分空間には恒等で作用し、一重項に位相 :math:`e^{i\theta}`
を与えます。:math:`\theta = \pi` で厳密なSWAP、:math:`\theta = \pi/2` で
（位相を除き）平方根SWAPです。全回路で ``exch(theta)[i, j]`` として使え、
両シミュレーションモードで動作し、autogradに完全対応します。

符号化
------

論理符号語は全スピン :math:`S = 1/2` セクターにあります
（:math:`|0_L\rangle` はスピン0, 1の一重項を使います）:

.. code-block:: python

   from blueqat.eo import encoding

   state = encoding.encode_state([(1, 0), (0, 1)])   # 6スピン上の |0>_L |1>_L
   encoding.leakage(state, triple=0)                 # 符号空間外への漏れ
   encoding.logical_action(u8x8)                     # 3スピンユニタリの2x2論理ブロック

各符号語は2つの *ゲージ* コピー（全Sz :math:`\pm 1/2`）を持ちます。
同梱の全パルス系列は、どちらのゲージセクターでも — 位相まで含めて —
同一に作用します。

論理回路からパルスへのトランスパイル
------------------------------------

:mod:`blueqat.eo` をインポートすると ``'eo'`` バックエンドが登録されます:

.. code-block:: python

   import blueqat.eo
   from blueqat import Circuit

   physical = Circuit(2).h[0].cx[0, 1].run(backend='eo')
   # 6スピン上の31交換パルス: H = 3パルス、Fong-Wandzura CNOT = 28パルス

Fong-Wandzura CNOTは直列28パルスの最近接系列です
（Weinstein et al., *Nature* **615**, 817 (2023)）。論理RZは1パルス、
X/Hは3パルスです。生成された回路は ``exch`` ゲートのみで構成され、任意の
シミュレーションバックエンドで実行できます:

.. code-block:: python

   init = encoding.encode_state([(1, 0), (1, 0)])
   final = physical.run(initial=init)      # 符号化Bell状態、忠実度1.0

微分可能なパルス合成
--------------------

シミュレータがtorchネイティブなので、パルス系列を勾配降下で *最適化*
できます:

.. code-block:: python

   from blueqat.eo import synthesize_1q, synthesize_2q, quantize_sequence

   # 任意のSU(2)を振幅一定の4パルスで合成 (忠実度 > 1 - 1e-9)
   seq = synthesize_1q(target_2x2, n_pulses=4)

   # ドリフトした2量子ビット系列を厳密なゲートに再校正。
   # ゲージ独立性は4つの全Szセクターにわたって強制されます
   refined = synthesize_2q(cx_4x4, pairs=pulse_pairs, initial_thetas=drifted)

   # パルス面積をハードウェアのクロック刻みに離散化
   seq_q = quantize_sequence(seq, step=2 * 3.141592653589793 / 4096)

パルススケジュール
------------------

:func:`~blueqat.eo.to_schedule` はパルスをJSON互換の時間分解スケジュールに
変換します（ASAP並列パッキング: 互いに素なスピン対のパルスは同時実行、
共有スピンの順序は保存されるためユニタリは不変です）:

.. code-block:: python

   from blueqat.eo import to_schedule, from_schedule, schedule_stats

   sched = to_schedule(physical)
   schedule_stats(sched)
   # {'n_pulses': 31, 'serial_duration': 94.2, 'scheduled_duration': 58.7,
   #  'parallel_speedup': 1.6}
   from_schedule(sched)      # Circuitに復元、ユニタリ保存

スケジュール形式はパルスレベル制御スタックや :doc:`クラウドバックエンド
<cloud>` 経由の送信を想定して設計されています。

トポロジーに関する注意
----------------------

生成されるパルスは、ゲートに関与するトリプル内の任意のペアにパルスを
打てることを仮定しています。厳密な最近接結合のみのハードウェアへの
マッピングには、ドット方位の割当てとスピンレベルのSWAPルーティングが
さらに必要で、これは今後の課題です。
