blueqat ドキュメント（日本語）
==============================

**blueqat** は PyTorch をネイティブ基盤とするオープンソースの量子計算SDKです。
回路は微分可能な状態ベクトル／テンソルネットワークシミュレータ上で実行され、
量子プログラムの勾配（VQE・QAOA・パルス最適化など）が autograd を通して
そのまま得られます。

.. code-block:: python

   from blueqat import Circuit

   # Bellペアを100ショットサンプリング
   Circuit(2).h[0].cx[0, 1].m[:].run(shots=100)
   # => Counter({'00': 52, '11': 48})

特徴
----

- **2つの実行モード** を同一APIで: 密な ``statevector`` と、大規模回路向けの
  ``tensornet`` （テンソルネットワーク縮約、デフォルト）。
- **端から端まで微分可能**: ゲートパラメータに ``requires_grad=True`` の
  ``torch.Tensor`` を渡せます。
- **Exchange-Onlyスピン量子ビット** (:mod:`blueqat.eo`): 3スピンに論理量子
  ビットを符号化し、回路をハイゼンベルク交換パルスへコンパイル。微分可能な
  パルス合成とハードウェア向けパルススケジュールも提供。
- **相互運用**: OpenQASM 2.0 の入出力、バージョン付きJSONシリアライズ、
  回路描画。
- **クラウド基盤** (:mod:`blueqat.cloud`): APIキー管理と ``backend='cloud'``
  での送信経路。

.. toctree::
   :maxdepth: 2
   :caption: ユーザーガイド

   getting_started
   guide/circuits
   guide/backends
   guide/autograd
   guide/exchange_only
   guide/cloud

APIリファレンス
---------------

APIリファレンスは :doc:`英語版 </api/index>` を参照してください
（docstringから自動生成されます）。

.. note::

   English documentation is :doc:`here </index>` /
   英語版ドキュメントは :doc:`こちら </index>`。
