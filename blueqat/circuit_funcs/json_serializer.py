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
"""Defines JSON serializer and deserializer for Blueqat circuits."""

import typing
from typing import Any, Dict, List, TypedDict, Union

from blueqat import Circuit
from blueqat.gate import Measurement, Operation
from ..gateset import create
from .flatten import flatten

SCHEMA_NAME = 'blueqat-circuit'
AVAILABLE_SCHEMA_VERSIONS = ["1", "2"]
LATEST_SCHEMA_VERSION = "2"


class SchemaJsonDict(TypedDict):
    """Schema header for detecting data type and version."""
    name: str
    version: str


class OpJsonDictV1(TypedDict):
    """Data type of Operation in Schema V1."""
    name: str
    params: List[float]
    targets: List[int]


class OpJsonDictV2(TypedDict):
    """Data type of Operation in Schema V2."""
    name: str
    params: List[float]
    options: Dict[str, Any]
    targets: List[int]


class CircuitJsonDictV1(TypedDict):
    """Data type of Circuit in Schema V1."""
    schema: SchemaJsonDict
    n_qubits: int
    ops: List[OpJsonDictV1]


class CircuitJsonDictV2(TypedDict):
    """Data type of Circuit in Schema V2."""
    schema: SchemaJsonDict
    n_qubits: int
    ops: List[OpJsonDictV2]


CircuitJsonDict = Union[CircuitJsonDictV1, CircuitJsonDictV2]


def serialize(c: Circuit) -> CircuitJsonDictV2:
    """Serialize Circuit into JSON-compatible dictionary.

    In this implementation, the serialized circuit is automatically flattened
    to break down multi-target operations into atomic gates.
    """
    def serialize_op(op: Operation) -> OpJsonDictV2:
        targets = op.targets
        if isinstance(targets, slice):
            raise TypeError('Circuit must be flattened before serialization.')
        if isinstance(targets, int):
            targets_list = [targets]
        elif isinstance(targets, tuple):
            targets_list = list(targets)
        else:
            targets_list = list(targets)  # fallback for other iterables

        options: dict[str, Any] = {}
        if isinstance(op, Measurement):
            if op.key is not None:
                options['key'] = op.key
            if op.duplicated is not None:
                options['duplicated'] = op.duplicated

        return {
            'name': str(op.lowername),
            'params': [float(p) for p in op.params],
            'options': options,
            'targets': targets_list
        }

    c_flat = flatten(c)
    return {
        'schema': {
            'name': SCHEMA_NAME,
            'version': LATEST_SCHEMA_VERSION
        },
        'n_qubits': c_flat.n_qubits,
        'ops': [serialize_op(op) for op in c_flat.ops]
    }


def deserialize(data: CircuitJsonDict) -> Circuit:
    """Deserialize JSON-compatible dictionary back into a Circuit object."""
    def make_op(opdata: Union[OpJsonDictV1, OpJsonDictV2]) -> Operation:
        return create(
            opdata['name'],
            tuple(opdata['targets']),
            tuple(float(p) for p in opdata['params']),
            opdata.get('options')
        )

    schema = data.get('schema', {})
    if schema.get('name', '') != SCHEMA_NAME:
        raise ValueError('Invalid schema name. This data is not a Blueqat circuit.')
        
    if schema.get('version', '') not in AVAILABLE_SCHEMA_VERSIONS:
        raise ValueError(f"Unknown schema version: {schema.get('version')}")
        
    n_qubits = data['n_qubits']
    ops = data['ops']
    return Circuit(n_qubits, [make_op(opdata) for opdata in ops])