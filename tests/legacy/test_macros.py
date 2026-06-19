from math import pi
import math
import cmath
import random
from collections import Counter

import pytest
import numpy as np

from blueqat import Circuit
from blueqat.circuit_funcs import circuit_to_unitary

import blueqat.macros

PAULI_X = np.array([[0, 1], [1, 0]])
ANGLES = [0.0, pi / 4, pi / 2, pi, -pi / 2, 2 * pi, -7 * pi]

ANGLES_SMALL = [0.0, -pi / 2, pi, 2 * pi]


def test_c3z():
    assert np.allclose(circuit_to_unitary(Circuit().c3z(0, 1, 2, 3)),
                       np.diag([1] * 15 + [-1]))


def test_c4z():
    assert np.allclose(circuit_to_unitary(Circuit().c4z(0, 1, 2, 3, 4)),
                       np.diag([1] * 31 + [-1]))


def test_mcz_gray_0():
    assert np.allclose(circuit_to_unitary(Circuit().mcz_gray([], 0)),
                       np.diag([1, -1]))


def test_mcz_gray_1():
    assert np.allclose(circuit_to_unitary(Circuit().mcz_gray([0], 1)),
                       np.diag([1, 1, 1, -1]))


def test_mcz_gray_4():
    assert np.allclose(circuit_to_unitary(Circuit().mcz_gray([0, 1, 2, 3], 4)),
                       np.diag([1] * 31 + [-1]))


def test_c3x():
    expected = np.eye(16)
    expected[7, 7] = expected[15, 15] = 0
    expected[7, 15] = expected[15, 7] = 1
    assert np.allclose(circuit_to_unitary(Circuit().c3x(0, 1, 2, 3)), expected)


def test_c4x():
    expected = np.eye(32)
    expected[15, 15] = expected[31, 31] = 0
    expected[15, 31] = expected[31, 15] = 1
    assert np.allclose(circuit_to_unitary(Circuit().c4x(0, 1, 2, 3, 4)),
                       expected)


def test_mcx_gray_0():
    assert np.allclose(circuit_to_unitary(Circuit().mcx_gray([], 0)),
                       np.array([[0, 1], [1, 0]]))


def test_mcx_gray_1():
    assert np.allclose(circuit_to_unitary(Circuit().mcx_gray([0], 1)),
                       circuit_to_unitary(Circuit().cx[0, 1]))


def test_mcx_gray_2():
    assert np.allclose(circuit_to_unitary(Circuit().mcx_gray([0, 1], 2)),
                       circuit_to_unitary(Circuit().ccx[0, 1, 2]))


def test_mcx_gray_4():
    expected = np.eye(32)
    expected[15, 15] = expected[31, 31] = 0
    expected[15, 31] = expected[31, 15] = 1
    assert np.allclose(circuit_to_unitary(Circuit().mcx_gray([0, 1, 2, 3], 4)),
                       expected)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_mcrx_gray_n(theta, n):
    u = circuit_to_unitary(Circuit().rx(theta)[0])
    expected = np.eye(2**(n + 1), dtype=complex)
    expected[2**n - 1, 2**n - 1] = u[0, 0]
    expected[2**(n + 1) - 1, 2**n - 1] = u[1, 0]
    expected[2**n - 1, 2**(n + 1) - 1] = u[0, 1]
    expected[2**(n + 1) - 1, 2**(n + 1) - 1] = u[1, 1]
    assert np.allclose(
        circuit_to_unitary(Circuit().mcrx_gray(theta, list(range(n)), n)),
        expected)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_mcry_gray_n(theta, n):
    u = circuit_to_unitary(Circuit().ry(theta)[0])
    expected = np.eye(2**(n + 1), dtype=complex)
    expected[2**n - 1, 2**n - 1] = u[0, 0]
    expected[2**(n + 1) - 1, 2**n - 1] = u[1, 0]
    expected[2**n - 1, 2**(n + 1) - 1] = u[0, 1]
    expected[2**(n + 1) - 1, 2**(n + 1) - 1] = u[1, 1]
    assert np.allclose(
        circuit_to_unitary(Circuit().mcry_gray(theta, list(range(n)), n)),
        expected)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_mcrz_gray_n(theta, n):
    u = circuit_to_unitary(Circuit().rz(theta)[0])
    expected = np.eye(2**(n + 1), dtype=complex)
    expected[2**n - 1, 2**n - 1] = u[0, 0]
    expected[2**(n + 1) - 1, 2**n - 1] = u[1, 0]
    expected[2**n - 1, 2**(n + 1) - 1] = u[0, 1]
    expected[2**(n + 1) - 1, 2**(n + 1) - 1] = u[1, 1]
    assert np.allclose(
        circuit_to_unitary(Circuit().mcrz_gray(theta, list(range(n)), n)),
        expected)


@pytest.mark.parametrize("theta", ANGLES)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_mcrz_gray_n(theta, n):
    u = circuit_to_unitary(Circuit().r(theta)[0])
    expected = np.eye(2**(n + 1), dtype=complex)
    expected[2**n - 1, 2**n - 1] = u[0, 0]
    expected[2**(n + 1) - 1, 2**n - 1] = u[1, 0]
    expected[2**n - 1, 2**(n + 1) - 1] = u[0, 1]
    expected[2**(n + 1) - 1, 2**(n + 1) - 1] = u[1, 1]
    assert np.allclose(
        circuit_to_unitary(Circuit().mcr_gray(theta, list(range(n)), n)),
        expected)


@pytest.mark.parametrize("theta", ANGLES_SMALL)
@pytest.mark.parametrize("phi", ANGLES_SMALL)
@pytest.mark.parametrize("lam", ANGLES_SMALL)
@pytest.mark.parametrize("gamma", ANGLES_SMALL)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_mcu_gray_n(theta, phi, lam, gamma, n):
    u = circuit_to_unitary(Circuit().u(theta, phi, lam, gamma)[0])
    expected = np.eye(2**(n + 1), dtype=complex)
    expected[2**n - 1, 2**n - 1] = u[0, 0]
    expected[2**(n + 1) - 1, 2**n - 1] = u[1, 0]
    expected[2**n - 1, 2**(n + 1) - 1] = u[0, 1]
    expected[2**(n + 1) - 1, 2**(n + 1) - 1] = u[1, 1]
    assert np.allclose(
        circuit_to_unitary(Circuit().mcu_gray(theta, phi, lam, gamma, list(range(n)), n)),
        expected)


def test_mcx_with_ancilla_0():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([], 1, 0)),
            circuit_to_unitary(Circuit().x[1])
            )


def test_mcx_with_ancilla_1():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1], 2, 0)),
            circuit_to_unitary(Circuit().cx[1, 2])
            )


def test_mcx_with_ancilla_2():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1, 2], 3, 0)),
            circuit_to_unitary(Circuit().ccx[1, 2, 3])
            )


def test_mcx_with_ancilla_3():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1, 2, 3], 4, 0)),
            circuit_to_unitary(Circuit().c3x(1, 2, 3, 4))
            )


def test_mcx_with_ancilla_4():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1, 2, 3, 4], 5, 0)),
            circuit_to_unitary(Circuit().c4x(1, 2, 3, 4, 5))
            )


def test_mcx_with_ancilla_5():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1, 2, 3, 4, 5], 6, 0)),
            circuit_to_unitary(Circuit().mcx_gray([1, 2, 3, 4, 5], 6))
            )


def test_mcx_with_ancilla_6():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcx_with_ancilla([1, 2, 3, 4, 5, 6], 7, 0)),
            circuit_to_unitary(Circuit().mcx_gray([1, 2, 3, 4, 5, 6], 7))
            )


def test_mcz_with_ancilla_0():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([], 1, 0)),
            circuit_to_unitary(Circuit().z[1])
            )


def test_mcz_with_ancilla_1():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1], 2, 0)),
            circuit_to_unitary(Circuit().cz[1, 2])
            )


def test_mcz_with_ancilla_2():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1, 2], 3, 0)),
            circuit_to_unitary(Circuit().ccz[1, 2, 3])
            )


def test_mcz_with_ancilla_3():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1, 2, 3], 4, 0)),
            circuit_to_unitary(Circuit().c3z(1, 2, 3, 4))
            )


def test_mcz_with_ancilla_4():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1, 2, 3, 4], 5, 0)),
            circuit_to_unitary(Circuit().mcz_gray([1, 2, 3, 4], 5))
            )


def test_mcz_with_ancilla_5():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1, 2, 3, 4, 5], 6, 0)),
            circuit_to_unitary(Circuit().mcz_gray([1, 2, 3, 4, 5], 6))
            )


def test_mcz_with_ancilla_6():
    assert np.allclose(
            circuit_to_unitary(Circuit().mcz_with_ancilla([1, 2, 3, 4, 5, 6], 7, 0)),
            circuit_to_unitary(Circuit().mcz_gray([1, 2, 3, 4, 5, 6], 7))
            )


def test_mcx_gray_12():
    cnt = Circuit().x[:12].mcx_gray(range(12), 12).m[:].shots(10)
    assert cnt == Counter({'1' * 13: 10})


def test_mcx_with_ancilla_12():
    cnt = Circuit().x[:12].mcx_with_ancilla(range(12), 12, 13).m[:].shots(10)
    assert cnt == Counter({'1' * 13 + '0': 10})
