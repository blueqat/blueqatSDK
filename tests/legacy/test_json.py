import json
import numpy as np
from blueqat import Circuit
from blueqat.circuit_funcs.json_serializer import serialize, deserialize
from blueqat.circuit_funcs.flatten import flatten


def test_json_dump_load():
    """A Circuit and deserialize(serialize())ed circuit returns same result.
    (However, it doesn't means they're same Circuit.)"""
    c = Circuit().h[:3].x[4].u(1.2, 3.4, 2.3, 1.0)[0].h[:].u(1.2, 3.4, 2.3)[1]
    d = serialize(c)
    j = json.dumps(d)
    d2 = json.loads(j)
    c2 = deserialize(d2)
    np.allclose(c.run(), c2.run())


def test_serialize_idempotent():
    """Serialized circuit and serialize(deserialize(serialize()))ed circuit are same."""
    c0 = Circuit().h[:3].x[4].cx[1, 3].u(1.2, 3.4, 2.3, 1.0)[0]
    c0.h[:].u(1.2, 3.4, 2.3)[1:].m[:]
    d1 = serialize(c0)
    c1 = deserialize(d1)
    d2 = serialize(c1)
    c2 = deserialize(d2)
    assert d1 == d2
    assert repr(c1) == repr(c2)


def test_serialize():
    """Testing serialized result. This JSON may changed in the future."""
    c = Circuit().r(0.2)[0, 2].h[:].m[1]
    d = serialize(c)
    assert d == {
        'schema': {
            'name': 'blueqat-circuit',
            'version': '2'
        },
        'n_qubits': 3,
        'ops': [
            {
                'name': 'phase',
                'targets': [0],
                'params': [0.2],
                'options': {}
            },
            {
                'name': 'phase',
                'targets': [2],
                'params': [0.2],
                'options': {}
            },
            {
                'name': 'h',
                'targets': [0],
                'params': [],
                'options': {}
            },
            {
                'name': 'h',
                'targets': [1],
                'params': [],
                'options': {}
            },
            {
                'name': 'h',
                'targets': [2],
                'params': [],
                'options': {}
            },
            {
                'name': 'measure',
                'targets': [1],
                'params': [],
                'options': {}
            },
        ]
    }


def test_deserialize():
    """Testing deserialize result. This JSON may changed in the future."""
    s = """{
    "schema": {"name": "blueqat-circuit", "version": "1"},
    "n_qubits": 3,
    "ops": [
      {"name": "phase", "targets": [0], "params": [0.2]},
      {"name": "phase", "targets": [2], "params": [0.2]},
      {"name": "h", "targets": [0], "params": []},
      {"name": "h", "targets": [1], "params": []},
      {"name": "h", "targets": [2], "params": []},
      {"name": "measure", "targets": [1], "params": []}
    ]}"""
    d = json.loads(s)
    c1 = deserialize(d)
    c2 = flatten(Circuit().r(0.2)[0, 2].h[:].m[1])
    assert repr(c1) == repr(c2)


def test_deserialize_unflatten():
    """Testing deserialize unflatten JSON file."""
    s = """{
    "schema": {"name": "blueqat-circuit", "version": "1"},
    "n_qubits": 3,
    "ops": [
      {"name": "phase", "targets": [0, 2], "params": [0.2]},
      {"name": "h", "targets": [0, 1, 2], "params": []},
      {"name": "measure", "targets": [1], "params": []}
    ]}"""
    d = json.loads(s)
    c1 = deserialize(d)
    c2 = Circuit().r(0.2)[0, 2].h[0, 1, 2].m[1]
    assert repr(c1) == repr(c2)


def test_serializeV2():
    """Testing serialized result. This JSON may changed in the future."""
    c = Circuit().m(key="foo")[0].h[:].m(key="foo", duplicated="replace")[1]
    d = serialize(c)
    assert d == {
        'schema': {
            'name': 'blueqat-circuit',
            'version': '2'
        },
        'n_qubits': 2,
        'ops': [
            {
                'name': 'measure',
                'targets': [0],
                'params': [],
                'options': {'key': 'foo'}
            },
            {
                'name': 'h',
                'targets': [0],
                'params': [],
                'options': {}
            },
            {
                'name': 'h',
                'targets': [1],
                'params': [],
                'options': {}
            },
            {
                'name': 'measure',
                'targets': [1],
                'params': [],
                'options': {'key': 'foo', 'duplicated': 'replace'}
            },
        ]
    }


def test_deserializeV1():
    """Testing deserialize result. This JSON may changed in the future."""
    s = """{
    "schema": {"name": "blueqat-circuit", "version": "1"},
    "n_qubits": 3,
    "ops": [
      {"name": "phase", "targets": [0], "params": [0.2]},
      {"name": "phase", "targets": [2], "params": [0.2]},
      {"name": "h", "targets": [0], "params": []},
      {"name": "h", "targets": [1], "params": []},
      {"name": "h", "targets": [2], "params": []},
      {"name": "measure", "targets": [1], "params": []}
    ]}"""
    d = json.loads(s)
    c1 = deserialize(d)
    c2 = flatten(Circuit().r(0.2)[0, 2].h[:].m[1])


