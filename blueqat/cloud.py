# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Groundwork for API-key based access to the Blueqat cloud service.

Credential resolution order:
1. An explicit `configure(api_key=...)` call in the current process.
2. The `BLUEQAT_API_KEY` environment variable.
3. The config file `~/.blueqat/config.json` (written by `save_api_key`,
   created with owner-only permissions).

The `cloud` backend registered by this module serializes a circuit to the
JSON wire format (see `blueqat.circuit_funcs.json_serializer`) and hands it
to a transport. Until the public endpoint is live, the default transport
raises a clear error; tests and early integrations can inject their own
transport with `configure(transport=...)`.

Importing this module registers the backend, so after `import blueqat.cloud`
a circuit can be submitted with `Circuit(...).run(backend='cloud')`.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .backends.backendbase import Backend, register_backend
from .gate import Operation

DEFAULT_ENDPOINT = "https://cloudapi.blueqat.com/v2"
ENV_API_KEY = "BLUEQAT_API_KEY"

_session: Dict[str, Any] = {"api_key": None, "endpoint": None, "transport": None}


def config_path() -> Path:
    """Path of the persistent config file (override dir with BLUEQAT_CONFIG_DIR)."""
    base = os.environ.get("BLUEQAT_CONFIG_DIR")
    root = Path(base) if base else Path.home() / ".blueqat"
    return root / "config.json"


def _load_config_file() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_api_key(api_key: str, endpoint: Optional[str] = None) -> Path:
    """Persist the API key to the config file with owner-only permissions."""
    if not api_key or not isinstance(api_key, str):
        raise ValueError("api_key must be a non-empty string.")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_config_file()
    data["api_key"] = api_key
    if endpoint is not None:
        data["endpoint"] = endpoint
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    # API keys are secrets: restrict the file to its owner.
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def delete_api_key() -> None:
    """Remove the stored API key from the config file (if present)."""
    path = config_path()
    data = _load_config_file()
    if "api_key" in data:
        del data["api_key"]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def get_api_key() -> Optional[str]:
    """Resolve the API key: configure() > environment > config file."""
    if _session["api_key"]:
        return _session["api_key"]
    env = os.environ.get(ENV_API_KEY)
    if env:
        return env
    return _load_config_file().get("api_key")


def get_endpoint() -> str:
    """Resolve the service endpoint: configure() > config file > default."""
    if _session["endpoint"]:
        return _session["endpoint"]
    return _load_config_file().get("endpoint", DEFAULT_ENDPOINT)


def configure(api_key: Optional[str] = None, endpoint: Optional[str] = None,
              transport: Optional[Callable[[Dict[str, Any]], Any]] = None) -> None:
    """Set session-level cloud settings (highest priority, not persisted).

    `transport` is a callable receiving the JSON-compatible request dict and
    returning the job result; inject one for tests or early integrations."""
    if api_key is not None:
        _session["api_key"] = api_key
    if endpoint is not None:
        _session["endpoint"] = endpoint
    if transport is not None:
        _session["transport"] = transport


def reset_configuration() -> None:
    """Clear session-level settings set by `configure` (env/file are untouched)."""
    _session["api_key"] = None
    _session["endpoint"] = None
    _session["transport"] = None


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "***"


class CloudBackend(Backend):
    """Backend submitting circuits to the Blueqat cloud service.

    The request payload is the versioned JSON circuit schema plus run
    parameters, so server and SDK can evolve independently."""

    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError(
                "Blueqat cloud API key is not set. Set the BLUEQAT_API_KEY "
                "environment variable, call blueqat.cloud.save_api_key(...), or "
                "blueqat.cloud.configure(api_key=...).")

        from .circuit import Circuit
        from .circuit_funcs.json_serializer import serialize
        request = {
            "circuit": serialize(Circuit(n_qubits, list(gates))),
            "shots": kwargs.get("shots"),
            "returns": kwargs.get("returns"),
            "options": {k: v for k, v in kwargs.items()
                        if k not in ("shots", "returns")},
        }

        transport = _session["transport"]
        if transport is None:
            raise RuntimeError(
                f"The Blueqat cloud service endpoint ({get_endpoint()}) is not "
                "available yet in this SDK version. Inject a transport with "
                "blueqat.cloud.configure(transport=...) to submit jobs.")
        return transport(request)

    def __repr__(self) -> str:
        key = get_api_key()
        status = f"api_key={_mask(key)}" if key else "unconfigured"
        return f"CloudBackend({status}, endpoint={get_endpoint()!r})"


# Importing blueqat.cloud makes the backend available as backend='cloud'.
register_backend("cloud", CloudBackend, overwrite=True)
