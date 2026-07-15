クラウドアクセス
================

:mod:`blueqat.cloud` は、Blueqatクラウドサービスへの APIキーベース
アクセスの基盤を提供します。

APIキー
-------

認証情報は次の優先順位で解決されます:

1. 現在のプロセスでの ``blueqat.cloud.configure(api_key=...)`` 呼び出し
2. 環境変数 ``BLUEQAT_API_KEY``
3. 設定ファイル ``~/.blueqat/config.json``

.. code-block:: python

   import blueqat.cloud as cloud

   cloud.save_api_key("YOUR_API_KEY")   # 所有者のみ読み書き可 (0600) で保存
   cloud.get_api_key()                  # 解決されたキー (ログ出力されず、reprではマスク)
   cloud.delete_api_key()

回路の送信
----------

:mod:`blueqat.cloud` をインポートすると ``'cloud'`` バックエンドが登録
されます。送信される回路は、実行パラメータとともにバージョン付きJSON
スキーマにシリアライズされます:

.. code-block:: python

   import blueqat.cloud
   from blueqat import Circuit

   Circuit(2).h[0].cx[0, 1].m[:].run(backend='cloud', shots=100)

公開エンドポイントが稼働するまで、デフォルトのトランスポートは明確な
エラーを送出します。テストや先行統合では独自トランスポートを注入できます:

.. code-block:: python

   def my_transport(request: dict):
       # request = {"circuit": {...}, "shots": 100, "returns": None, "options": {...}}
       return {"job_id": "...", "status": "queued"}

   blueqat.cloud.configure(api_key="...", transport=my_transport)
