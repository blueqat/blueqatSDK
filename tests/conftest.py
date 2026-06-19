import os
import sys

# 1. 探索パスの最優先登録
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

# 2. メモリ残骸のクリーニング
for mod_name in list(sys.modules.keys()):
    if mod_name == "blueqat" or mod_name.startswith("blueqat."):
        del sys.modules[mod_name]

# 3. 新生バックエンドをテスト空間に強制注入 💡 (ここを追記)
from blueqat.backends import BACKENDS
from blueqat.backends.torch_backend import TorchBackend

BACKENDS["statevector"] = lambda: TorchBackend(mode="statevector")
BACKENDS["torch"] = lambda: TorchBackend(mode="statevector")
BACKENDS["tensornet"] = lambda: TorchBackend(mode="tensornet")
BACKENDS["torch_tn"] = lambda: TorchBackend(mode="tensornet")

print("\n✨ [pytest] TorchBackends registered successfully!")