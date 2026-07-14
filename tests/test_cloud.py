"""Tests for the API-key based cloud-access groundwork (blueqat.cloud).

No network access happens anywhere here: the transport is injectable, and the
default transport refuses to run until the public endpoint ships.
"""
import json
import os
import stat

import pytest

import blueqat.cloud as cloud
from blueqat import Circuit


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config file at a temp dir and clear env/session state."""
    monkeypatch.setenv("BLUEQAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(cloud.ENV_API_KEY, raising=False)
    cloud.reset_configuration()
    yield
    cloud.reset_configuration()


# --- credential resolution -------------------------------------------------------

def test_no_key_by_default():
    assert cloud.get_api_key() is None


def test_env_var_key(monkeypatch):
    monkeypatch.setenv(cloud.ENV_API_KEY, "env-key-123")
    assert cloud.get_api_key() == "env-key-123"


def test_save_and_load_key_file():
    path = cloud.save_api_key("file-key-456")
    assert cloud.get_api_key() == "file-key-456"
    data = json.loads(path.read_text())
    assert data["api_key"] == "file-key-456"


def test_config_file_permissions_owner_only():
    path = cloud.save_api_key("secret-key")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == (stat.S_IRUSR | stat.S_IWUSR), f"config file mode is {oct(mode)}"


def test_configure_beats_env_and_file(monkeypatch):
    cloud.save_api_key("file-key")
    monkeypatch.setenv(cloud.ENV_API_KEY, "env-key")
    cloud.configure(api_key="session-key")
    assert cloud.get_api_key() == "session-key"


def test_env_beats_file(monkeypatch):
    cloud.save_api_key("file-key")
    monkeypatch.setenv(cloud.ENV_API_KEY, "env-key")
    assert cloud.get_api_key() == "env-key"


def test_delete_api_key():
    cloud.save_api_key("to-be-deleted")
    cloud.delete_api_key()
    assert cloud.get_api_key() is None


def test_endpoint_resolution():
    assert cloud.get_endpoint() == cloud.DEFAULT_ENDPOINT
    cloud.save_api_key("k", endpoint="https://staging.example.com")
    assert cloud.get_endpoint() == "https://staging.example.com"
    cloud.configure(endpoint="https://session.example.com")
    assert cloud.get_endpoint() == "https://session.example.com"


def test_save_rejects_empty_key():
    with pytest.raises(ValueError):
        cloud.save_api_key("")


# --- backend wiring -----------------------------------------------------------------

def test_cloud_backend_reachable_from_circuit_run():
    # Importing blueqat.cloud registers the 'cloud' backend in the plugin
    # registry, which Circuit.run consults.
    with pytest.raises(RuntimeError, match="API key is not set"):
        Circuit(1).h[0].run(backend="cloud")


def test_cloud_backend_requires_transport_when_key_set():
    cloud.configure(api_key="some-key")
    with pytest.raises(RuntimeError, match="not.*available yet"):
        Circuit(1).h[0].run(backend="cloud")


def test_cloud_backend_submits_serialized_circuit():
    captured = {}

    def fake_transport(request):
        captured.update(request)
        return {"job_id": "job-1", "status": "queued"}

    cloud.configure(api_key="some-key", transport=fake_transport)
    result = Circuit(2).h[0].cx[0, 1].m[:].run(backend="cloud", shots=100)

    assert result == {"job_id": "job-1", "status": "queued"}
    assert captured["shots"] == 100
    circuit_json = captured["circuit"]
    assert circuit_json["schema"]["name"] == "blueqat-circuit"
    assert circuit_json["n_qubits"] == 2
    names = [op["name"] for op in circuit_json["ops"]]
    assert names == ["h", "cx", "measure", "measure"]


def test_cloud_backend_repr_masks_key():
    cloud.configure(api_key="super-secret-api-key-value")
    r = repr(cloud.CloudBackend())
    assert "super-secret-api-key-value" not in r
    assert "supe" in r  # masked prefix only
