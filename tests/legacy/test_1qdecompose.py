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

from math import pi
import random
from typing import Callable, List

import pytest
import torch
import numpy as np

from blueqat import Circuit
from blueqat.gate import OneQubitGate, Mat1Gate, HGate, UGate, PhaseGate, RXGate, RYGate, RZGate
from blueqat.circuit_funcs import circuit_to_unitary
from blueqat.backends.onequbitgate_decomposer import ryrz_decomposer, u_decomposer

Decomposer = Callable[[OneQubitGate], List[OneQubitGate]]

decomposer_test = pytest.mark.parametrize('decomposer',
                                          [ryrz_decomposer, u_decomposer])


def to_tensor(val) -> torch.Tensor:
    """NumPy配列やリスト、既存のTensorを確実に complex128 の torch.Tensor に変換する"""
    if isinstance(val, torch.Tensor):
        return val.to(dtype=torch.complex128)
    if isinstance(val, np.ndarray):
        return torch.from_numpy(val).to(dtype=torch.complex128)
    return torch.tensor(val, dtype=torch.complex128)


def check_decomposed(g: OneQubitGate, d: Decomposer, ignore_global: bool):
    c1 = Circuit(1, [g])
    c2 = Circuit(1, d(g))
    
    # 💡 戻り値が NumPy 配列であっても確実に Tensor にラップする
    u1 = to_tensor(circuit_to_unitary(c1))
    u2 = to_tensor(circuit_to_unitary(c2))
    
    if ignore_global:
        gphase1 = torch.det(u1).angle()
        gphase2 = torch.det(u2).angle()
        
        su1 = u1 * torch.exp(-0.5j * gphase1)
        su2 = u2 * torch.exp(-0.5j * gphase2)
        
        assert torch.isclose(torch.det(su1), torch.tensor(1.0+0j, dtype=torch.complex128), atol=1e-4)
        assert torch.isclose(torch.det(su2), torch.tensor(1.0+0j, dtype=torch.complex128), atol=1e-4)
    else:
        su1 = su2 = torch.eye(2, dtype=torch.complex128)
        
    try:
        if ignore_global:
            assert torch.allclose(su1, su2, atol=1e-4) or torch.allclose(su1, -su2, atol=1e-4)
        else:
            assert torch.allclose(u1, u2, atol=1e-4)
    except AssertionError:
        print("Orig:", c1)
        print(u1)
        if ignore_global:
            print("-->")
            print(su1)
        print("Conv:", c2)
        print(u2)
        if ignore_global:
            print("-->")
            print(su2)
            print("abs(Orig - Conv):")
            print(torch.abs(su1 - su2))
            print("abs(Orig + Conv):")
            print(torch.abs(su1 + su2))
        else:
            print("abs(Orig - Conv):")
            print(torch.abs(u1 - u2))
        raise


@decomposer_test
def test_identity(decomposer):
    g = Mat1Gate((0, ), torch.eye(2, dtype=torch.complex128))
    check_decomposed(g, decomposer, False)


@decomposer_test
def test_identity_plus_delta(decomposer):
    g_matrix = torch.eye(2, dtype=torch.complex128) + torch.ones((2, 2), dtype=torch.complex128) * 1e-10
    g = Mat1Gate((0, ), g_matrix)
    check_decomposed(g, decomposer, False)


@decomposer_test
def test_hadamard(decomposer):
    g = HGate((0, ))
    check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_rx(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RXGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_ry(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RYGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_rz(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = RZGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_r(decomposer):
    for _ in range(20):
        t = random.random() * pi
        g = PhaseGate((0, ), t)
        check_decomposed(g, decomposer, True)


@decomposer_test
def test_random_u(decomposer):
    for _ in range(20):
        t1, t2, t3, t4 = [random.random() * pi for _ in range(4)]
        g = UGate((0, ), t1, t2, t3, t4)
        check_decomposed(g, decomposer, True)