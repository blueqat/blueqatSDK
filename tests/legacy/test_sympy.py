# Copyright 2019 The Blueqat Developers
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

from functools import reduce

import numpy as np
from sympy import eye, zeros, symbols, simplify, sin, cos, exp, pi, sqrt, I, Matrix, nfloat
from sympy.physics.quantum import TensorProduct

from blueqat import Circuit

def test_sympy_backend_for_one_qubit_gate():
    E = eye(2)
    X = Matrix([[0, 1], [1, 0]])
    Y = Matrix([[0, -I], [I, 0]])
    Z = Matrix([[1, 0], [0, -1]])
    H = Matrix([[1, 1], [1, -1]]) / sqrt(2)
    T = Matrix([[1, 0], [0, exp(I*pi/4)]])
    S = Matrix([[1, 0], [0, exp(I*pi/2)]])

    x, y, z = symbols('x, y, z')
    RX = Matrix([[cos(x / 2), -I * sin(x / 2)], [-I * sin(x / 2), cos(x / 2)]])
    RY = Matrix([[cos(y / 2), -sin(y / 2)], [sin(y / 2), cos(y / 2)]])
    RZ = Matrix([[cos(z / 2) - I * sin(z / 2), 0], [0, cos(z / 2) + I * sin(z / 2)]])

    actual_1 = Circuit().x[0, 1].y[1].z[2].run(backend="sympy_unitary")
    expected_1 = reduce(TensorProduct, [Z, Y * X, X])
    assert actual_1 == expected_1

    actual_2 = Circuit().y[0].z[3].run(backend="sympy_unitary")
    expected_2 = reduce(TensorProduct, [Z, E, E, Y])
    assert actual_2 == expected_2

    actual_3 = Circuit().x[0].z[3].h[:].t[1].s[2].run(backend="sympy_unitary")
    expected_3 = reduce(TensorProduct, [H * Z, S * H, T * H, H * X])
    assert actual_3 == expected_3

    actual_4 = Circuit().rx(-np.pi / 2)[0].rz(np.pi / 2)[1].ry(np.pi)[2].run(backend="sympy_unitary")
    expected_4 = reduce(TensorProduct, [RY, RZ, RX]).subs([[x, -pi / 2], [y, pi], [z, pi / 2]])
    assert actual_4 == expected_4


def test_sympy_backend_for_two_qubit_gate1():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    X = Matrix([[0, 1], [1, 0]])
    Z = Matrix([[1, 0], [0, -1]])
    H = Matrix([[1, 1], [1, -1]]) / sqrt(2)
    H_3 = reduce(TensorProduct, [H, E, H])
    H_4 = reduce(TensorProduct, [H, E, E, H])
    CX_3 = reduce(TensorProduct, [E, E, UPPER]) + reduce(TensorProduct, [X, E, LOWER])
    CZ_3 = reduce(TensorProduct, [E, E, UPPER]) + reduce(TensorProduct, [Z, E, LOWER])
    CX_4 = reduce(TensorProduct, [E, E, E, UPPER]) + reduce(TensorProduct, [X, E, E, LOWER])
    CZ_4 = reduce(TensorProduct, [E, E, E, UPPER]) + reduce(TensorProduct, [Z, E, E, LOWER])

    actual_1 = Circuit().cx[0, 3].run(backend="sympy_unitary")
    assert actual_1 == CX_4

    actual_2 = Circuit().cx[1, 3].x[4].run(backend="sympy_unitary")
    expected_2 = reduce(TensorProduct, [X, CX_3, E])
    assert actual_2 == expected_2

    actual_3 = Circuit().cz[0, 3].run(backend="sympy_unitary")
    assert actual_3 == CZ_4

    actual_4 = Circuit().cz[1, 3].x[4].run(backend="sympy_unitary")
    expected_4 = reduce(TensorProduct, [X, CZ_3, E])
    assert actual_4 == expected_4

    actual_5 = Circuit().cx[3, 0].run(backend="sympy_unitary")
    assert actual_5 == H_4 * CX_4 * H_4

    actual_6 = Circuit().cx[3, 1].x[4].run(backend="sympy_unitary")
    assert actual_6 == reduce(TensorProduct, [X, H_3 * CX_3 * H_3, E])

    actual_7 = Circuit().cz[3, 0].run(backend="sympy_unitary")
    assert actual_7 == CZ_4

    actual_8 = Circuit().cz[3, 1].x[4].run(backend="sympy_unitary")
    assert actual_8 == reduce(TensorProduct, [X, CZ_3, E])


def test_sympy_backend_for_two_qubit_gate2():
    x, y, z = symbols('x y z', real=True)
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    RX = Matrix([[cos(x / 2), -I * sin(x / 2)], [-I * sin(x / 2), cos(x / 2)]])
    RY = Matrix([[cos(y / 2), -sin(y / 2)], [sin(y / 2), cos(y / 2)]])
    RZ = Matrix([[cos(z / 2) - I * sin(z / 2), 0], [0, cos(z / 2) + I * sin(z / 2)]])
    CRX_3 = reduce(TensorProduct, [E, UPPER, E]) + reduce(TensorProduct, [E, LOWER, RX])
    CRY_4 = reduce(TensorProduct, [UPPER, E, E]) + reduce(TensorProduct, [LOWER, RY, E])
    CRZ_3 = reduce(TensorProduct, [E, E, UPPER]) + reduce(TensorProduct, [RZ, E, LOWER])

    actual_9 = Circuit().crx(x)[1, 0].i[2].run(backend="sympy_unitary")
    assert simplify(actual_9 - CRX_3) == zeros(8)

    actual_10 = Circuit().cry(y)[2, 1].run(backend="sympy_unitary")
    assert simplify(actual_10 - CRY_4) == zeros(8)

    actual_11 = Circuit().crz(z)[0, 2].run(backend="sympy_unitary")
    m = simplify(nfloat(actual_11) - nfloat(CRZ_3))


def test_sympy_cx_cz():
    assert Circuit().cx[1, 2].run(backend="sympy_unitary") == Circuit().h[2].cz[2, 1].h[2].run(backend="sympy_unitary")
    assert Circuit().cx[2, 1].run(backend="sympy_unitary") == Circuit().h[1].cz[2, 1].h[1].run(backend="sympy_unitary")


def test_u():
    theta, phi, lambd = symbols("theta phi lambd")

    actual_1 = simplify(Circuit().u(theta, phi, lambd, -(phi + lambd) / 2)[0].run(backend="sympy_unitary"))
    assert actual_1[0, 0] != 0
    expected_1 = simplify(Circuit().rz(lambd)[0].ry(theta)[0].rz(phi)[0].run_with_sympy_unitary())
    assert expected_1[0, 0] != 0
    assert simplify(nfloat(actual_1)) == simplify(nfloat(expected_1))


def test_cr():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    lambd = symbols("lambd")
    U = Circuit().r(lambd)[0].run_with_sympy_unitary()
    U /= U[0, 0]

    actual_1 = Circuit().cr(lambd)[0, 1].run(backend="sympy_unitary")
    expected_1 = reduce(TensorProduct, [UPPER, E]) + reduce(TensorProduct, [LOWER, U])
    assert simplify(actual_1 - expected_1) == zeros(4)


def test_cu():
    E = eye(2)
    UPPER = Matrix([[1, 0], [0, 0]])
    LOWER = Matrix([[0, 0], [0, 1]])
    theta, phi, lambd = symbols("theta phi lambd")
    U = Circuit().rz(lambd)[0].ry(theta)[0].rz(phi)[0].run_with_sympy_unitary()

    actual_1 = Circuit().cu(theta, phi, lambd, -(phi + lambd) / 2)[0, 1].run(backend="sympy_unitary")
    expected_1 = reduce(TensorProduct, [E, UPPER]) + reduce(TensorProduct, [U, LOWER])
    print("actual")
    print(simplify(actual_1))
    print("expected")
    print(simplify(expected_1))
    print("diff")
    print(simplify(actual_1 - expected_1))
    assert simplify(actual_1 - expected_1) == zeros(4)


def test_toffoli_sympy():
    assert simplify(Circuit().ccx[2, 1, 0].to_unitary(ignore_global=True)) == Matrix([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]])


def test_chgate():
    u = simplify(Circuit().ch[1, 0].to_unitary())
    assert simplify(u - Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1/sqrt(2), 1/sqrt(2)],
        [0, 0, 1/sqrt(2), -1/sqrt(2)]])) == zeros(4)


def test_cygate():
    u1 = Circuit().cy[1, 0].to_unitary()
    u2 = Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -I],
        [0, 0, I, 0]])
    assert simplify(u1 - u2) == zeros(4)


def test_rxgate():
    t = symbols('t')
    u = Circuit().rx(t)[0].to_unitary()
    expected = exp(-I * t / 2 * Circuit().x[0].to_unitary())
    assert simplify(u - expected) == zeros(2)


def test_rygate():
    t = symbols('t')
    u = Circuit().ry(t)[0].to_unitary()
    expected = simplify(exp(-I * t / 2 * Circuit().y[0].to_unitary()))
    assert simplify(u - expected) == zeros(2)


def test_rzgate():
    t = symbols('t')
    u = Circuit().rz(t)[0].to_unitary()
    expected = exp(-I * t / 2 * Circuit().z[0].to_unitary())
    assert simplify(u - expected) == zeros(2)


def test_rxxgate():
    t = symbols('t')
    u = Circuit().rxx(t)[1, 0].to_unitary()
    expected = simplify(exp(-I * t / 2 * Circuit().x[0, 1].to_unitary()))
    assert simplify(u - expected) == zeros(4)


def test_ryygate():
    t = symbols('t')
    u = Circuit().ryy(t)[1, 0].to_unitary()
    expected = simplify(exp(-I * t / 2 * Circuit().y[0, 1].to_unitary()))
    assert simplify(u - expected) == zeros(4)


def test_rzzgate():
    t = symbols('t')
    u = Circuit().rzz(t)[1, 0].to_unitary()
    expected = simplify(exp(-I * t / 2 * Circuit().z[0, 1].to_unitary()))
    assert simplify(u - expected) == zeros(4)


def test_swapgate():
    assert simplify(Circuit().swap[1, 0].to_unitary() -
                    Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) == zeros(4)


def test_cswapgate():
    expected = eye(8)
    expected[4:, 4:] = Circuit().swap[1, 0].to_unitary()
    assert simplify(Circuit().cswap[2, 1, 0].to_unitary() - expected) == zeros(8)
