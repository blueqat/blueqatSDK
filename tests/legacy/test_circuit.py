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

import math
import cmath
import random
from collections import Counter

import pytest
import numpy as np

from blueqat import Circuit, BlueqatGlobalSetting
from blueqat.backends.onequbitgate_decomposer import u_decomposer
from blueqat.utils import ignore_global_phase

EPS = 1e-16


def vec_distsq(a, b):
    diff = a - b
    return diff.T.conjugate() @ diff


def test_hgate1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().h[1].h[0].run(backend=backend, 
                                               shots=shots),
                       np.array([0.5, 0.5, 0.5, 0.5]))


def test_hgate2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().x[0].h[0].run(backend=backend,
                                               shots=shots),
                       np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]))


def test_hgate3(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().h[:2].run(backend=backend,
                                           shots=shots),
                       Circuit().h[0].h[1].run(backend=backend,
                                               shots=shots))


def test_pauli1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        ignore_global_phase(Circuit().x[0].y[0].run(backend=backend,
                                                    shots=shots)),
        Circuit().z[0].run(backend=backend,
                           shots=shots))


def test_pauli2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        ignore_global_phase(Circuit().y[0].z[0].run(backend=backend,
                                                    shots=shots)),
        Circuit().x[0].run(backend=backend,
                           shots=shots))


def test_pauli3(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().z[0].x[0].run(backend=backend,
                                shots=shots),
        ignore_global_phase(Circuit().y[0].run(backend=backend,
                                               shots=shots)))


def test_cx1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().h[0].h[1].cx[1, 0].h[0].h[1].run(backend=backend,
                                                   shots=shots),
        Circuit().cx[0, 1].run(backend=backend,
                               shots=shots))


def test_cx2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().x[2].cx[:4:2, 1:4:2].run(backend=backend,
                                                          shots=shots),
                       Circuit().x[2:4].run(backend=backend,
                                            shots=shots))


def test_cx3(backend):
    '''Refer issues #76 (https://github.com/Blueqat/Blueqat/issues/76)'''
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit().z[2].x[0].cx[0, 1]
    assert np.allclose(c.run(backend=backend,
                             shots=shots),
                       np.array([0., 0., 0., 1., 0., 0., 0., 0.]))


def test_cx4(backend):
    '''Refer issues #76 (https://github.com/Blueqat/Blueqat/issues/76)'''
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit(4).x[0].cx[0, 1]
    result = np.zeros(16)
    result[3] = 1.0
    assert np.allclose(c.run(backend=backend,
                             shots=shots), result)


def test_cx5(backend):
    '''Refer issues #76 (https://github.com/Blueqat/Blueqat/issues/76)'''
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit(4).x[0].cx[0, 2].cx[0, 1]
    result = np.zeros(16)
    result[7] = 1.0
    assert np.allclose(c.run(backend=backend,
                             shots=shots), result)


def test_cx6(backend):
    '''Refer issues #76 (https://github.com/Blueqat/Blueqat/issues/76)'''
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit().x[0].cx[0, 3].cx[3, 1]
    result = np.zeros(16)
    result[11] = 1.0
    assert np.allclose(c.run(backend=backend,
                             shots=shots), result)


def test_rz1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        ignore_global_phase(Circuit().h[0].rz(
            math.pi)[0].run(backend=backend,
                            shots=shots)),
        Circuit().x[0].h[0].run(backend=backend,
                                shots=shots))
    if backend not in ['quimb']:
        assert np.allclose(Circuit().h[0].r(math.pi)[0].run(backend=backend,
                                                            shots=shots),
                        Circuit().x[0].h[0].run(backend=backend,
                                                shots=shots))


def test_rz2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().h[0].rz(math.pi / 3)[0].h[1].rz(math.pi /
                                                  3)[1].run(backend=backend,
                                                            shots=shots),
        Circuit().h[0, 1].rz(math.pi / 3)[:].run(backend=backend,
                                                 shots=shots))


def test_tgate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().t[0].run(backend=backend, shots=shots),
        ignore_global_phase(Circuit().rz(math.pi / 4)[0].run(backend=backend,
                                                             shots=shots)))


def test_sgate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().s[0].run(backend=backend, shots=shots),
        ignore_global_phase(Circuit().rz(math.pi / 2)[0].run(backend=backend,
                                                             shots=shots)))


def test_tdg_gate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        ignore_global_phase(Circuit().s[1].tdg[1].tdg[1].run(backend=backend,
                                                             shots=shots)),
        Circuit().i[1].run(backend=backend,
                           shots=shots))


def test_sdg_gate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().s[1].sdg[1].run(backend=backend,
                                                 shots=shots),
                       Circuit().i[1].run(backend=backend,
                                          shots=shots))


def test_y_gate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().y[1].run(backend=backend,
                                          shots=shots),
                       Circuit().z[1].x[1].run(backend=backend,
                                               shots=shots) * 1j)


@pytest.mark.parametrize('bits', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_toffoli_gate(bits, backend):
    c = Circuit()
    if bits[0]:
        c.x[0]
    if bits[1]:
        c.x[1]
    if backend in ['quimb']:
        res_counter = c.ccx[0, 1, 2].run(backend=backend, shots=1)
        expected_meas = "1" if bits[0] and bits[1] else "0"
        assert list(res_counter)[0][2] == expected_meas
    else:
        c.ccx[0, 1, 2].m[2]
        expected_meas = "001" if bits[0] and bits[1] else "000"
        assert c.run(backend=backend, shots=1) == Counter([expected_meas])


def test_u_gate(backend):
    if backend in ['quimb']:
        shots = -1
        assert np.allclose(
            Circuit().u(1.23, 4.56, -5.43)[1].run(backend=backend,
                                                    shots=shots),
            ignore_global_phase(
            Circuit().rz(-5.43)[1].ry(1.23)[1].rz(4.56)[1].run(backend=backend,
                                                            shots=shots)))
    else:
        shots = None
        assert np.allclose(
            Circuit().u(1.23, 4.56, -5.43,
                        -0.5 * (4.56 - 5.43))[1].run(backend=backend,
                                                    shots=shots),
            Circuit().rz(-5.43)[1].ry(1.23)[1].rz(4.56)[1].run(backend=backend,
                                                            shots=shots))


def test_r_gate(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(Circuit().r(-1.23)[1].run(backend=backend,
                                                 shots=shots),
                       Circuit().u(0, 0, -1.23)[1].run(backend=backend,
                                                       shots=shots))


def test_rotation1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().ry(-math.pi / 2)[0].rz(math.pi / 6)[0].ry(
            math.pi / 2)[0].run(backend=backend,
                                shots=shots),
        Circuit().rx(math.pi / 6)[0].run(backend=backend,
                                         shots=shots))


def test_crotation(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().cu(1.23, 4.56, -5.43,
                     -0.5 * (4.56 - 5.43))[3, 1].run(backend=backend,
                                                     shots=shots),
        Circuit().crz(-5.43)[3,
                             1].cry(1.23)[3,
                                          1].crz(4.56)[3,
                                                       1].run(backend=backend,
                                                              shots=shots))


def test_crotation2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    assert np.allclose(
        Circuit().crx(1.23)[1, 3].run(backend=backend,
                                      shots=shots),
        Circuit().h[3].crz(1.23)[1, 3].h[3].run(backend=backend,
                                                shots=shots))


def test_crotation3(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    p0 = math.acos(math.sqrt(0.3)) * 2
    p1 = math.acos(math.sqrt(0.4)) * 2
    # |00> -> 0.3
    # |10> -> 0.7 * 0.4 = 0.28
    # |01> -> 0.0
    # |11> -> 0.7 * 0.6 = 0.42
    assert np.allclose(Circuit().ry(p0)[0].cry(p1)[0, 1].run(backend=backend,
                                                             shots=shots),
                       np.sqrt(np.array([0.3, 0.28, 0.0, 0.42])))


def test_globalphase_rz(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    c = Circuit().rz(theta)[0]
    assert abs(c.run(backend=backend,
                     shots=shots)[0] - cmath.exp(-0.5j * theta)) < 1e-8


def test_globalphase_r(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    c = Circuit().phase(theta)[0]
    assert abs(c.run(backend=backend,
                     shots=shots)[0] - 1) < 1e-8


def test_globalphase_u(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    phi = 1.6
    lambd = 2.3
    c = Circuit().u(theta, phi, lambd)[0]
    assert abs(c.run(backend=backend,
                     shots=shots)[0] - math.cos(theta / 2)) < 1e-8


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_globalphase_u_with_gamma(backend):
    theta = 1.2
    phi = 1.6
    lambd = 2.3
    gamma = -1.4
    c = Circuit().u(theta, phi, lambd, gamma)[0]
    assert abs(
        c.run(backend=backend)[0] -
        cmath.exp(1j * gamma) * math.cos(theta / 2)) < 1e-8


def test_controlled_gate_phase_crz(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    val = np.exp(-0.5j * theta)
    c0 = Circuit().crz(theta)[0, 1]
    c1 = Circuit().x[0] + c0
    v0 = c0.run(backend=backend,
                shots=shots)
    v1 = c1.run(backend=backend,
                shots=shots)
    assert abs(abs(v0[0]) - 1) < 1e-8
    assert abs(v1[1] / v0[0] - val) < 1e-8


def test_controlled_gate_phase_cphase(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    val = 1.0
    c0 = Circuit().cphase(theta)[0, 1]
    c1 = Circuit().x[0] + c0
    v0 = c0.run(backend=backend,
                shots=shots)
    v1 = c1.run(backend=backend,
                shots=shots)
    assert abs(abs(v0[0]) - 1) < 1e-8
    assert abs(v1[1] / v0[0] - val) < 1e-8


def test_controlled_gate_phase_cu(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    theta = 1.2
    phi = 1.6
    lambd = 2.3
    c0 = Circuit().cu(theta, phi, lambd)[0, 1]
    c1 = Circuit().x[0] + c0
    v0 = c0.run(backend=backend,
                shots=shots)
    v1 = c1.run(backend=backend,
                shots=shots)
    assert abs(abs(v0[0]) - 1) < 1e-8
    assert abs(v1[1] / v0[0] - math.cos(theta / 2)) < 1e-8


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_controlled_gate_phase_cu_with_gamma(backend):
    theta = 1.2
    phi = 1.6
    lambd = 2.3
    gamma = 1.4

    c0 = Circuit().cu(theta, phi, lambd, gamma)[0, 1]
    c1 = Circuit().x[0] + c0
    v0 = c0.run(backend=backend)
    v1 = c1.run(backend=backend)
    assert abs(abs(v0[0]) - 1) < 1e-8
    assert abs(v1[1] / v0[0] -
               cmath.exp(1j * gamma) * math.cos(theta / 2)) < 1e-8


def test_measurement1(backend):
    c = Circuit().m[0]
    cnt = c.run(backend=backend, shots=100)
    assert cnt.most_common(1) == [("0", 100)]


def test_measurement2(backend):
    c = Circuit().x[0].m[0]
    cnt = c.run(backend=backend, shots=100)
    assert cnt.most_common(1) == [("1", 100)]


def test_measurement3(backend):
    # 75% |0> + 25% |1>
    c = Circuit().rx(math.pi / 3)[0].m[0]
    n = 500
    cnt = c.run(backend=backend, shots=n)
    most_common = cnt.most_common(1)[0]
    assert most_common[0] == "0"
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 3σ = 3 * sqrt(np(1-p))
    three_sigma = 3 * np.sqrt(n * 0.75 * 0.25)
    assert abs(most_common[1] - 0.75 * n) < three_sigma


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_measurement_multiqubit1(backend):
    c = Circuit().x[0].m[1]
    cnt = c.run(backend=backend, shots=100)
    # 0-th qubit is also 0 because it is not measured.
    assert cnt.most_common(1) == [("00", 100)]


def test_measurement_multiqubit2(backend):
    c = Circuit().x[0].m[1::-1]
    cnt = c.run(backend=backend, shots=100)
    assert cnt.most_common(1) == [("10", 100)]


def test_measurement_entangled_state(backend):
    # 1/sqrt(2) (|0> + |1>)
    c = Circuit().h[0].cx[0, 1]
    for _ in range(100):
        cnt = c.run(backend=backend, shots=1)
        result = cnt.most_common()
        assert result == [("00", 1)] or result == [("11", 1)]


def test_measurement_hadamard1(backend):
    n = 1000
    c = Circuit().h[0].m[0]
    cnt = c.run(backend=backend, shots=n)
    a, b = cnt.most_common(2)
    assert a[1] + b[1] == n
    # variance of binomial distribution (n -> ∞) is np(1-p)
    # therefore, 3σ = 3 * sqrt(np(1-p))
    three_sigma = 3 * np.sqrt(n * 0.5 * 0.5)
    assert abs(a[1] - n / 2) < three_sigma


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_measurement_after_qubits1(backend):
    for _ in range(50):
        c = Circuit().h[0].m[0]
        a, cnt = c.run(backend=backend,
                       shots=1,
                       returns="statevector_and_shots")
        if cnt.most_common(1)[0] == ('0', 1):
            assert np.allclose(a, np.array([1, 0]))
        else:
            assert np.allclose(a, np.array([0, 1]))


def test_caching_then_expand(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit().h[0]
    c.run(backend=backend,
          shots=shots)
    qubits = c.i[1].run(backend=backend,
                        shots=shots)
    assert np.allclose(qubits, Circuit().h[0].i[1].run(backend=backend, shots=shots))


def test_copy_empty(backend):
    if backend in ['quimb']:
        c = Circuit(1)
    else:
        c = Circuit()
    c.run(backend=backend)
    cc = c.copy(copy_backends=True)
    assert c.ops == cc.ops and c.ops is not cc.ops
    if backend in ['numpy', 'numba']:
        assert c._backends[backend].cache is None and cc._backends[
            backend].cache is None
        assert c._backends[backend].cache_idx == cc._backends[
            backend].cache_idx == -1


def test_cache_then_append(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit()
    c.x[0]
    c.run()
    c.h[0]
    c.run()
    assert np.allclose(c.run(backend=backend,
                             shots=shots),
                       Circuit().x[0].h[0].run(backend=backend,
                                               shots=shots))


def test_concat_circuit1(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c1 = Circuit()
    c1.h[0]
    c1.run()
    c2 = Circuit()
    c2.h[1]
    c2.run()
    c1 += c2
    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                       Circuit().h[0].h[1].run(backend=backend,
                                               shots=shots))


def test_concat_circuit2(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c1 = Circuit()
    c1.h[1]
    c1.run()
    c2 = Circuit()
    c2.h[0]
    c2.run()
    c1 += c2
    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                       Circuit().h[1].h[0].run(backend=backend,
                                               shots=shots))


def test_concat_circuit3(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c1 += c2
    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                       Circuit().x[0].h[0].run(backend=backend,
                                               shots=shots))
    c1 = Circuit()
    c1.h[0]
    c2 = Circuit()
    c2.x[0]
    c1 += c2
    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                       Circuit().h[0].x[0].run(backend=backend,
                                               shots=shots))


def test_concat_circuit4(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c1 = Circuit()
    c1.x[0]
    c2 = Circuit()
    c2.h[0]
    c = c1 + c2
    c.run()
    assert np.allclose(c.run(backend=backend,
                             shots=shots),
                       Circuit().x[0].h[0].run(backend=backend,
                                               shots=shots))
    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                       Circuit().x[0].run(backend=backend,
                                          shots=shots))
    assert np.allclose(c2.run(backend=backend,
                              shots=shots),
                       Circuit().h[0].run(backend=backend,
                                          shots=shots))


def test_complicated_circuit(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    c = Circuit()
    c.x[0].h[0].rx(-1.5707963267948966)[2].cx[0, 2].rz(0.095491506289)[2]
    c.cx[0, 2].h[0].rx(1.5707963267948966)[2].h[0].ry(-1.5707963267948966)[2]
    c.cx[0, 2].cx[2, 3].rz(0.095491506289)[3].cx[2, 3].cx[0, 2].h[0]
    c.rx(1.5707963267948966)[2].h[0].ry(-1.5707963267948966)[2].cx[0, 1]
    c.cx[1,
         2].rz(0.095491506289)[2].cx[1,
                                     2].cx[0,
                                           1].h[0].u(math.pi / 2, 1.58,
                                                     -0.62)[2]
    c.h[0].rx(-1.5707963267948966)[2].cx[0, 1].cx[1, 2].cx[2, 3]
    c.rz(0.095491506289)[3].cx[2, 3].cx[1, 2].cx[0, 1].h[0]
    c.rx(1.5707963267948966)[2].u(0.42, -1.5707963267948966, 1.64)[2].h[2]
    c.cx[0, 2].rz(-0.095491506289)[2].cx[0, 2].rx(1.5707963267948966)[0].h[2]
    c.rx(-1.5707963267948966)[0].h[2].cx[0, 2].cx[2, 3].rz(-0.095491506289)[3]
    c.cx[2, 3].cx[0, 2].rx(1.5707963267948966)[0].h[2]
    c.rx(-1.5707963267948966)[0].h[2].cx[0, 1].cx[1, 2].rz(-0.095491506289)[2]
    c.cx[1, 2].cx[0, 1].rx(1.5707963267948966)[0].h[2]
    c.rx(-1.5707963267948966)[0].t[2].s[2].cx[0, 1].cx[1, 2].sdg[1].cx[2, 3]
    c.rz(-0.095491506289)[3].cx[2, 3].cx[1, 2].cx[0, 1]
    c.rx(1.5707963267948966)[0].h[2].h[0].rx(-1.5707963267948966)[1].h[2]
    c.cx[0, 1].cx[1, 2].rz(1.1856905316303521e-08)[2].cx[1, 2].cx[0, 1].h[0]
    c.rx(1.5707963267948966)[1].h[2].rx(-1.5707963267948966)[0]
    c.rx(-1.5707963267948966)[1].rx(-1.5707963267948966)[2].cx[0, 1].cx[1, 2]
    c.rz(1.1856905316303521e-08)[2].cx[1, 2].cx[0, 1].rx(1.5707963267948966)[0]
    c.rx(1.5707963267948966)[1].rx(1.5707963267948966)[2]
    c.rx(-1.5707963267948966)[1].cx[1, 3].rz(1.2142490216037756e-08)[3]
    c.cx[1, 3].rx(1.5707963267948966)[1].rx(-1.5707963267948966)[1].cx[0, 1]
    c.cx[1, 2].rz(-1.2142490216037756e-08)[2].cx[1, 2].cx[0, 1]
    c.rx(1.5707963267948966)[1]
    c.cx[0, 3].u(0.23, 1.24, -0.65)[3].cx[3, 1].cx[3, 0]
    vec = c.run(backend=backend,shots=shots)
    assert np.allclose(
        ignore_global_phase(vec),
        np.array([
            5.88423813e-01 + 0.00000000e+00j, -3.82057626e-02 -
            5.70122617e-02j, -2.52821022e-17 - 5.09095967e-17j,
            -1.21188626e-11 + 5.63063568e-10j, -2.19604047e-01 -
            2.85449458e-01j, -2.59211189e-03 + 4.58219688e-02j,
            3.08617333e-09 - 3.56619861e-09j, 4.48946755e-18 - 3.62425819e-19j,
            4.64439684e-09 - 1.48402425e-09j, 4.61321871e-18 - 4.67197922e-18j,
            -3.59382904e-01 + 4.73135946e-01j, 2.20759589e-02 +
            6.42836440e-02j, -1.55912415e-17 - 3.57403200e-17j,
            5.05381446e-10 + 2.03362289e-10j, 3.82475330e-01 - 1.07620677e-01j,
            2.29456407e-02 - 3.47003613e-02j
        ]))


def test_switch_backend1():
    c = Circuit().x[0].h[0]
    _backend = BlueqatGlobalSetting.get_default_backend_name()
    if _backend in ['quimb']:
        assert np.array_equal(c.run(shots=-1), c.run(backend="numpy"))
    else:
        assert np.array_equal(c.run(), c.run(backend="numpy"))

    BlueqatGlobalSetting.set_default_backend("qasm_output")
    assert c.run() == c.to_qasm()

    # Different instance of QasmOutputBackend is used.
    # Lhs is owned by Circuit, rhs is passed as argument. But in this case same result.
    from blueqat.backends.qasm_output_backend import QasmOutputBackend
    assert c.run(output_prologue=False) == c.run(False,
                                                 backend=QasmOutputBackend())

    BlueqatGlobalSetting.set_default_backend("numpy")
    assert c.run(shots=5) == c.run_with_numpy(shots=5)


def test_macro():
    def macro(c, i):
        return c.h[i]

    BlueqatGlobalSetting.register_macro('foo', macro)
    try:
        assert np.allclose(Circuit().foo(1).run(), Circuit().h[1].run())
    finally:
        BlueqatGlobalSetting.unregister_macro('foo')


@pytest.mark.parametrize('pair', [
    (Circuit(10).x[:].reset[:], Circuit(10)),
    (Circuit(10).h[:].reset[:], Circuit(10)),
])
def test_reset1(backend, pair):
    if backend in ['qgate', 'quimb']:
        pytest.xfail('mat1 gate for this backend is unimplemented.')
    assert np.allclose(pair[0].run(backend=backend),
                       pair[1].run(backend=backend))


def test_reset2(backend):
    if backend in ['qgate', 'quimb']:
        pytest.xfail('mat1 gate for this backend is unimplemented.')
    common = Circuit().h[0].cx[0, 1].cx[0, 2].reset[1].m[:].run(
        backend=backend, shots=100).most_common(3)
    assert len(common) == 2
    a, b = common
    assert a[0] in ('000', '101')
    assert b[0] in ('000', '101')


def test_mat1(backend):
    if backend in ['qgate', 'quimb']:
        pytest.xfail('mat1 gate for this backend is unimplemented.')
    p = random.random() * math.pi
    q = random.random() * math.pi
    r = random.random() * math.pi
    a1 = Circuit().u(p, q, r)[0].run(backend=backend)
    a2 = Circuit().x[0].u(p, q, r)[0].run(backend=backend)
    a = np.hstack([a1.reshape((2, 1)), a2.reshape((2, 1))])

    b1 = Circuit().mat1(a)[0].run(backend=backend)
    assert np.allclose(a1, b1)
    b2 = Circuit().x[0].mat1(a)[0].run(backend=backend)
    assert np.allclose(a2, b2)


def test_mat1_2(backend):
    if backend in ['qgate', 'quimb']:
        pytest.xfail('mat1 gate for this backend is unimplemented.')
    t = random.random() * math.pi
    a = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    v1 = Circuit().mat1(a)[1:3].run(backend=backend)
    v2 = Circuit().ry(t * 2)[1:3].run(backend=backend)
    assert np.allclose(v1, v2)


def test_swap(backend):
    if backend in ['quimb']:
        shots = -1
    else:
        shots = None
    p = random.random() * math.pi
    q = random.random() * math.pi
    r = random.random() * math.pi
    s = random.random() * math.pi

    c1 = Circuit().ry(p)[0].rz(q)[0].ry(r)[1].rz(s)[1]
    c2 = Circuit().ry(p)[1].rz(q)[1].ry(r)[0].rz(s)[0].swap[0, 1]

    assert np.allclose(c1.run(backend=backend,
                              shots=shots),
                        c2.run(backend=backend,
                                shots=shots))


def test_mat1_decomposite(backend):
    if backend in ['qgate', 'quimb']:
        pytest.xfail('mat1 gate for this backend is unimplemented.')
    p = random.random() * math.pi
    q = random.random() * math.pi
    r = random.random() * math.pi
    g = random.random() * math.pi
    a1 = Circuit().u(p, q, r, g)[0].run(backend=backend)
    a2 = Circuit().x[0].u(p, q, r, g)[0].run(backend=backend)
    a = np.hstack([a1.reshape((2, 1)), a2.reshape((2, 1))])

    c = Circuit().mat1(a)[2, 4]
    v1 = c.run(backend=backend)
    v2 = c.run_with_2q_decomposition(
        basis='cx', mat1_decomposer=u_decomposer).run(backend=backend)
    assert np.allclose(v1, v2)


@pytest.mark.parametrize('basis', ['cx', 'cz', 'zz'])
def test_decomposite1(basis):
    p = random.random()
    q = random.random()
    r = random.random()
    s = random.random()

    c = Circuit().ry(p)[1].rz(q)[1].ry(r)[3].rz(s)[3].cz[3, 1].h[2].ry(
        r)[3].rz(s)[3].ry(p)[1].rz(q)[1]
    v1 = c.run()
    v2 = c.run_with_2q_decomposition(basis=basis).run()
    assert np.allclose(ignore_global_phase(v1), ignore_global_phase(v2))


@pytest.mark.parametrize('basis', ['cx', 'cz', 'zz'])
def test_decomposite2(basis):
    p = random.random()
    q = random.random()
    r = random.random()
    s = random.random()

    c = Circuit().ry(p)[1].rz(q)[1].ry(r)[0].rz(s)[0].cx[0, 1].h[2].ry(
        r)[0].rz(s)[0].ry(p)[1].rz(q)[1]
    v1 = c.run()
    v2 = c.run_with_2q_decomposition(basis=basis).run()
    assert np.allclose(ignore_global_phase(v1), ignore_global_phase(v2))


@pytest.mark.parametrize('basis', ['cx', 'cz', 'zz'])
def test_decomposite3(basis):
    p = random.random()
    q = random.random()
    r = random.random()
    s = random.random()

    c = Circuit().ry(p)[1].rz(q)[1].ry(r)[0].rz(s)[0].zz[0, 1].h[2].ry(
        r)[0].rz(s)[0].ry(p)[1].rz(q)[1]
    v1 = c.run()
    v2 = c.run_with_2q_decomposition(basis=basis).run()
    assert np.allclose(ignore_global_phase(v1), ignore_global_phase(v2))


def test_initial_vec(backend):
    if backend == 'qgate':
        import qgate
        try:
            qgate.__version__
        except AttributeError:
            pytest.xfail("This version of qgate doesn't support initial vec.")
    if backend in ['quimb']:
        pytest.xfail("This backend doesn't support initial vec.")

    c = Circuit().h[0]
    v1 = c.run(backend=backend)
    assert np.allclose(c.run(backend=backend, initial=v1),
                       Circuit(1).run(backend=backend))


def test_initial_vec2(backend):
    if backend == 'qgate':
        import qgate
        try:
            qgate.__version__
        except AttributeError:
            pytest.xfail("This version of qgate doesn't support initial vec.")
    if backend in ['quimb']:
        pytest.xfail("This backend doesn't support initial vec.")

    v = Circuit().x[1].run(backend=backend)
    cnt = Circuit().x[0].m[0, 1].run(backend=backend, initial=v, shots=100)
    assert cnt == Counter({'11': 100})


def test_initial_vec3(backend):
    if backend == 'qgate':
        import qgate
        try:
            qgate.__version__
        except AttributeError:
            pytest.xfail("This version of qgate doesn't support initial vec.")
    if backend in ['quimb']:
        pytest.xfail("This backend doesn't support initial vec.")

    v = Circuit(4).h[3].run(backend=backend)
    v2 = Circuit(4).run(backend=backend, initial=v)
    assert np.allclose(v, v2)


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_statevector_method(backend):
    assert isinstance(
        Circuit().h[0].cx[0, 1].m[:].statevector(backend=backend), np.ndarray)


def test_shots_method(backend):
    assert isinstance(
        Circuit().h[0].cx[0, 1].m[:].shots(200, backend=backend), Counter)


@pytest.mark.skipif(
    condition=lambda backend: backend == 'quimb',
    reason='Skip test for specific backend',
)
def test_oneshot_method(backend):
    vec, meas = Circuit().h[0].cx[0, 1].m[:].oneshot(backend=backend)
    assert isinstance(vec, np.ndarray)
    assert isinstance(meas, str)
