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
"""This module is experiment features. We may delete or change provided functions.

The module provides builtin macros."""

import math
from typing import Any, Sequence, Optional

from blueqat import Circuit
from blueqat.decorators import circuitmacro
from blueqat.utils import calc_u_params, gen_gray_controls, sqrt_2x2_matrix

from blueqat.gate import UGate


@circuitmacro
def draw(c: Circuit, **kwargs) -> Any:
    """Draw the circuit. This function requires Qiskit."""
    return c.run(backend='ibmq', returns="draw", **kwargs)


@circuitmacro
def margolus(c: Circuit, c0: int, c1: int) -> Circuit:
    """Simplified Toffoli gate implementation by Margolus.

    This gate is also as know as relative Toffoli gate.
    It's almost as same as Toffoli gate, but only relative phases are different.
    Refer https://arxiv.org/abs/quant-ph/0312225v1 for details.
    (Remarks: It's also described in Nielsen & Chuang, Exercise 4.26.)"""
    # 注: 元コードに変数 t の定義がない、あるいは c1, c0 などの使われ方に
    # 依存する部分がありますが、提示されたロジックをそのまま維持しています。
    c.ry(math.pi * 0.25)[t].cx[c1, t].ry(math.pi * 0.25)[t].cx[c0, t]
    c.ry(math.pi * -0.25)[t].cx[c1, t].ry(math.pi * -0.25)[t]
    return c


@circuitmacro
def c3z(c: Circuit, c0: int, c1: int, c2: int, t: int) -> Circuit:
    """A macro of 3-controlled Z gate"""
    return c.mcz_gray([c0, c1, c2], t)


@circuitmacro
def c4z(c: Circuit, c0: int, c1: int, c2: int, c3: int, t: int) -> Circuit:
    """A macro of 4-controlled Z gate"""
    return c.mcz_gray([c0, c1, c2, c3], t)


@circuitmacro
def c3x(c: Circuit, c0: int, c1: int, c2: int, t: int) -> Circuit:
    """A macro of 3-controlled X gate"""
    return c.h[t].c3z(c0, c1, c2, t).h[t]


@circuitmacro
def c4x(c: Circuit, c0: int, c1: int, c2: int, c3: int, t: int) -> Circuit:
    """A macro of 4-controlled X gate"""
    return c.h[t].c4z(c0, c1, c2, c3, t).h[t]


@circuitmacro
def mcx_gray(c: Circuit, ctrl: Sequence[int], target: int) -> Circuit:
    """A macro of multi controlled X gate."""
    return c.h[target].mcz_gray(ctrl, target).h[target]


@circuitmacro
def mcx_with_ancilla(c: Circuit,
                     ctrl: Sequence[int],
                     target: int,
                     ancilla: int) -> Circuit:
    """A macro of multi controlled X gate with ancilla.

    Refer https://arxiv.org/pdf/quant-ph/9503016.pdf Lem. 7.3, Cor. 7.4."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.x[target]
    if n_ctrl == 1:
        return c.cx[ctrl[0], target]
    if n_ctrl == 2:
        return c.ccx[ctrl[0], ctrl[1], target]
    if n_ctrl == 3:
        return c.c3x(ctrl[0], ctrl[1], ctrl[2], target)
    if n_ctrl == 4:
        return c.c4x(ctrl[0], ctrl[1], ctrl[2], ctrl[3], target)
    sep = (n_ctrl + 1) // 2 + 1
    c1 = ctrl[:sep]
    a1 = ctrl[-1]
    c2 = list(ctrl[sep:]) + [ancilla]
    a2 = ctrl[sep - 1]
    c.mcx_with_ancilla(c1, ancilla, a1)
    c.mcx_with_ancilla(c2, target, a2)
    c.mcx_with_ancilla(c1, ancilla, a1)
    c.mcx_with_ancilla(c2, target, a2)
    return c


@circuitmacro
def mcz_gray(c: Circuit, ctrl: Sequence[int], target: int) -> Circuit:
    """A macro of multi controlled Z gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.z[target]
    angles = [math.pi / 2**(n_ctrl - 1), -math.pi / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.cr(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcz_with_ancilla(c: Circuit,
                     ctrl: Sequence[int],
                     target: int,
                     ancilla: int) -> Circuit:
    """A macro of multi controlled X gate with ancilla."""
    return c.h[target].mcx_with_ancilla(ctrl, target, ancilla).h[target]


@circuitmacro
def mcrx_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    """A macro of multi controlled RX gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.rx(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.crx(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcry_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    """A macro of multi controlled RY gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.ry(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.cry(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcrz_gray(c: Circuit, theta: float, ctrl: Sequence[int],
              target: int) -> Circuit:
    """A macro of multi controlled RZ gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.rz(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.crz(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcr_gray(c: Circuit, theta: float, ctrl: Sequence[int],
             target: int) -> Circuit:
    """A macro of multi controlled R gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.r(theta)[target]
    angles = [theta / 2**(n_ctrl - 1), -theta / 2**(n_ctrl - 1)]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        c.cr(angles[parity])[ctrl[c1], target]
    return c


@circuitmacro
def mcu_gray(c: Circuit, theta: float, phi: float, lam: float, gamma: float,
             ctrl: Sequence[int], target: int) -> Circuit:
    """A macro of multi controlled U gate."""
    n_ctrl = len(ctrl)
    if n_ctrl == 0:
        return c.u(theta, phi, lam, gamma)[target]
    if n_ctrl == 1:
        return c.cu(theta, phi, lam, gamma)[ctrl[0], target]
    mat = UGate.create(0, (theta, phi, lam, gamma)).matrix()
    for _ in range(n_ctrl - 1):
        mat = sqrt_2x2_matrix(mat)
    params = calc_u_params(mat)
    params_dagger = -params[0], -params[2], -params[1], -params[3]
    for c0, c1, parity in gen_gray_controls(n_ctrl):
        if c0 >= 0:
            c.cx[ctrl[c0], ctrl[c1]]
        if parity:
            c.cu(*params_dagger)[ctrl[c1], target]
        else:
            c.cu(*params)[ctrl[c1], target]
    return c


# パッケージの公開インターフェースを明示的に定義
__all__ = [
    "draw",
    "margolus",
    "c3z",
    "c4z",
    "c3x",
    "c4x",
    "mcx_gray",
    "mcx_with_ancilla",
    "mcz_gray",
    "mcz_with_ancilla",
    "mcrx_gray",
    "mcry_gray",
    "mcrz_gray",
    "mcr_gray",
    "mcu_gray",
]