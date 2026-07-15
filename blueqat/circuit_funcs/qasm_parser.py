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
"""Parser for a practical subset of OpenQASM 2.0 (the qelib1.inc gate set) into a Circuit.

This is the reverse of `Circuit.to_qasm()`.
"""

import ast
import math
import operator
import re
from typing import List

from ..circuit import Circuit

# name -> (blueqat gate name, number of numeric args)
_GATES = {
    # no-arg 1-qubit
    'x': ('x', 0), 'y': ('y', 0), 'z': ('z', 0), 'h': ('h', 0),
    's': ('s', 0), 'sdg': ('sdg', 0), 't': ('t', 0), 'tdg': ('tdg', 0),
    'sx': ('sx', 0), 'sxdg': ('sxdg', 0), 'id': ('i', 0), 'reset': ('reset', 0),
    # 1-arg 1-qubit
    'rx': ('rx', 1), 'ry': ('ry', 1), 'rz': ('rz', 1),
    'p': ('phase', 1), 'u1': ('phase', 1),
    # 3-arg 1-qubit
    'u': ('u', 3), 'u3': ('u', 3),
    # no-arg 2-qubit
    'cx': ('cx', 0), 'cz': ('cz', 0), 'cy': ('cy', 0), 'ch': ('ch', 0), 'swap': ('swap', 0),
    # 1-arg 2-qubit
    'cp': ('cphase', 1), 'cu1': ('cphase', 1),
    'crx': ('crx', 1), 'cry': ('cry', 1), 'crz': ('crz', 1),
    'rxx': ('rxx', 1), 'ryy': ('ryy', 1), 'rzz': ('rzz', 1),
    # blueqat's zz gate is the fixed diag(1, i, i, 1) -- it takes no angle
    'zz': ('zz', 0),
    # 4-arg 2-qubit
    'cu': ('cu', 4),
    # no-arg 3-qubit
    'ccx': ('ccx', 0), 'cswap': ('cswap', 0),
}

_CONSTS = {'pi': math.pi}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_angle_expr(node: ast.AST) -> float:
    """Safely evaluate a numeric angle expression (numbers, +-*/, unary -, and `pi`)
    without falling back to `eval`, since this parses (potentially untrusted) QASM text."""
    if isinstance(node, ast.Expression):
        return _eval_angle_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_angle_expr(node.left), _eval_angle_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_angle_expr(node.operand))
    raise ValueError(f"Unsupported expression in QASM angle argument: {ast.dump(node)}")


def _parse_args(args_str: str) -> List[float]:
    args = []
    for a in args_str.split(','):
        try:
            tree = ast.parse(a.strip(), mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Could not parse QASM argument {a!r}: {e}") from e
        args.append(_eval_angle_expr(tree))
    return args


def _parse_targets(targets_str: str) -> List[int]:
    return [int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', targets_str)]


def from_qasm(qasm: str) -> Circuit:
    """Parse an OpenQASM 2.0 program (the qelib1.inc gate set) into a Circuit."""
    c = Circuit()
    # strip line (//) and block (/* */) comments
    text = re.sub(r'/\*.*?\*/', '', qasm, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)

    for raw_stmt in text.split(';'):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        if stmt.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'gate ', 'opaque ')):
            continue

        barrier_match = re.match(r'barrier\s+(.+)$', stmt)
        if barrier_match:
            targets = _parse_targets(barrier_match.group(1))
            if targets:
                c.barrier[tuple(targets) if len(targets) > 1 else targets[0]]
            # "barrier q;" (whole register, no explicit indices) is dropped:
            # the register size isn't tracked here.
            continue

        measure_match = re.match(r'measure\s+q\[(\d+)\]\s*->\s*c\[(\d+)\]$', stmt)
        if measure_match:
            c.m[int(measure_match.group(1))]
            continue

        gate_match = re.match(r'(\w+)\s*(?:\(([^)]*)\))?\s+(.+)$', stmt)
        if not gate_match:
            raise ValueError(f"Could not parse QASM statement: {stmt!r}")
        name, args_str, targets_str = gate_match.groups()

        if name not in _GATES:
            raise ValueError(f"Unsupported QASM gate: {name!r}")
        gate_name, n_args = _GATES[name]
        args = _parse_args(args_str) if args_str else []
        if len(args) != n_args:
            raise ValueError(f"Gate {name!r} expects {n_args} argument(s), got {len(args)}")
        targets = _parse_targets(targets_str)

        op = getattr(c, gate_name)
        if args:
            op = op(*args)
        target = targets[0] if len(targets) == 1 else tuple(targets)
        op[target]

    return c
