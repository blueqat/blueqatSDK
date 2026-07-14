"""Standard quantum-computing correctness tests, in the style used by other
quantum SDKs (Qiskit / Cirq): algebraic gate identities, canonical entangled
states, the QFT against the DFT matrix, coherent teleportation, dagger
(uncompute) round-trips, cross-backend consistency, Trotterized time evolution
against the exact matrix exponential, and shot-statistics sanity checks.
"""
import cmath
import math
import random

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.utils import X, Y, Z, term_from_chars

import numpy as np

ATOL = 1e-8

MODES = ["statevector", "tensornet"]


def _u(c: Circuit) -> np.ndarray:
    return circuit_to_unitary(c)


def _allclose_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = ATOL) -> bool:
    tr = np.trace(a.conj().T @ b)
    if abs(tr) < atol:
        return False
    phase = tr / abs(tr)
    return np.allclose(a * phase, b, atol=atol)


# --- 1. Algebraic gate identities ------------------------------------------

IDENTITY_CASES = [
    ("HXH=Z", Circuit(1).h[0].x[0].h[0], Circuit(1).z[0]),
    ("HZH=X", Circuit(1).h[0].z[0].h[0], Circuit(1).x[0]),
    ("S^2=Z", Circuit(1).s[0].s[0], Circuit(1).z[0]),
    ("T^2=S", Circuit(1).t[0].t[0], Circuit(1).s[0]),
    ("T^4=Z", Circuit(1).t[0].t[0].t[0].t[0], Circuit(1).z[0]),
    ("SX^2=X", Circuit(1).sx[0].sx[0], Circuit(1).x[0]),
    ("X^2=I", Circuit(1).x[0].x[0], Circuit(1).i[0]),
    ("Y^2=I", Circuit(1).y[0].y[0], Circuit(1).i[0]),
    ("Z^2=I", Circuit(1).z[0].z[0], Circuit(1).i[0]),
    ("H^2=I", Circuit(1).h[0].h[0], Circuit(1).i[0]),
    ("S*Sdg=I", Circuit(1).s[0].sdg[0], Circuit(1).i[0]),
    ("T*Tdg=I", Circuit(1).t[0].tdg[0], Circuit(1).i[0]),
    ("SX*SXdg=I", Circuit(1).sx[0].sxdg[0], Circuit(1).i[0]),
    ("CX^2=I", Circuit(2).cx[0, 1].cx[0, 1], Circuit(2).i[0]),
    ("CZ^2=I", Circuit(2).cz[0, 1].cz[0, 1], Circuit(2).i[0]),
    ("SWAP^2=I", Circuit(2).swap[0, 1].swap[0, 1], Circuit(2).i[0]),
    ("CCX^2=I", Circuit(3).ccx[0, 1, 2].ccx[0, 1, 2], Circuit(3).i[0]),
    # XYZ = iI, so up to global phase it's the identity
    ("XYZ~I", Circuit(1).z[0].y[0].x[0], Circuit(1).i[0]),
    # CZ is symmetric in its qubits
    ("CZ symm", Circuit(2).cz[0, 1], Circuit(2).cz[1, 0]),
    # CX with H conjugation flips control/target
    ("HH-CX-HH", Circuit(2).h[0].h[1].cx[0, 1].h[0].h[1], Circuit(2).cx[1, 0]),
    # SWAP = 3 alternating CX
    ("SWAP=3CX", Circuit(2).cx[0, 1].cx[1, 0].cx[0, 1], Circuit(2).swap[0, 1]),
]


@pytest.mark.parametrize("case", IDENTITY_CASES, ids=lambda c: c[0])
def test_gate_identity(case):
    name, lhs, rhs = case
    assert _allclose_up_to_global_phase(_u(lhs), _u(rhs)), f"identity {name} violated"


def test_euler_decomposition_u_gate():
    # U(theta, phi, lam) == e^{i alpha} RZ(phi) RY(theta) RZ(lam)
    theta, phi, lam = 0.7, 1.1, -0.4
    u = _u(Circuit(1).u(theta, phi, lam)[0])
    zyz = _u(Circuit(1).rz(lam)[0].ry(theta)[0].rz(phi)[0])
    assert _allclose_up_to_global_phase(u, zyz)


def test_rz_pi_is_z_up_to_phase():
    assert _allclose_up_to_global_phase(_u(Circuit(1).rz(math.pi)[0]), _u(Circuit(1).z[0]))


def test_phase_gate_equals_rz_up_to_phase():
    theta = 0.9
    assert _allclose_up_to_global_phase(
        _u(Circuit(1).phase(theta)[0]), _u(Circuit(1).rz(theta)[0]))


# --- 2. Canonical entangled states ------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_bell_state(mode):
    state = Circuit(2).h[0].cx[0, 1].run(backend=mode)
    expected = torch.zeros(4, dtype=torch.complex128)
    expected[0] = expected[3] = 1 / math.sqrt(2)
    assert torch.allclose(state, expected, atol=ATOL)


@pytest.mark.parametrize("mode", MODES)
def test_ghz_state(mode):
    n = 5
    c = Circuit(n).h[0]
    for i in range(n - 1):
        c.cx[i, i + 1]
    state = c.run(backend=mode)
    expected = torch.zeros(2 ** n, dtype=torch.complex128)
    expected[0] = expected[-1] = 1 / math.sqrt(2)
    assert torch.allclose(state, expected, atol=ATOL)


def test_w_state_3qubit():
    # W = (|001> + |010> + |100>)/sqrt(3), built with RY + controlled ops.
    theta = 2 * math.acos(1 / math.sqrt(3))
    c = Circuit(3)
    c.ry(theta)[0]
    c.ch[0, 1]
    c.cx[1, 2].cx[0, 1].x[0]
    state = c.run().detach().numpy()
    expected = np.zeros(8, dtype=complex)
    expected[0b001] = expected[0b010] = expected[0b100] = 1 / math.sqrt(3)
    assert np.allclose(np.abs(state), np.abs(expected), atol=ATOL)


# --- 3. QFT vs the DFT matrix ------------------------------------------------

def _qft_circuit(n: int) -> Circuit:
    c = Circuit(n)
    for i in reversed(range(n)):
        c.h[i]
        for k in range(i):
            c.cphase(math.pi / 2 ** (i - k))[k, i]
    for i in range(n // 2):
        c.swap[i, n - 1 - i]
    return c


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_dft_matrix(n):
    dim = 2 ** n
    omega = cmath.exp(2j * math.pi / dim)
    dft = np.array([[omega ** (j * k) for j in range(dim)] for k in range(dim)]) / math.sqrt(dim)
    assert np.allclose(_u(_qft_circuit(n)), dft, atol=ATOL)


def test_qft_dagger_is_inverse():
    c = _qft_circuit(3)
    u = _u(c + c.dagger())
    assert np.allclose(u, np.eye(8), atol=ATOL)


# --- 4. Coherent quantum teleportation ---------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_coherent_teleportation(mode):
    # Teleport a random 1-qubit state from qubit 0 to qubit 2, replacing the
    # usual measurement + classical corrections with coherent CX/CZ corrections.
    rng = random.Random(42)
    alpha_angle, phase_angle = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)

    c = Circuit(3)
    c.ry(alpha_angle)[0].rz(phase_angle)[0]     # prepare |psi> on q0
    psi = Circuit(1).ry(alpha_angle)[0].rz(phase_angle)[0].run().detach().numpy()

    c.h[1].cx[1, 2]                             # Bell pair on q1, q2
    c.cx[0, 1].h[0]                             # Bell measurement basis rotation
    c.cx[1, 2].cz[0, 2]                         # coherent corrections

    state = c.run(backend=mode).detach().numpy()
    # Expected: |+>_0 |+>_1 |psi>_2 with qubit 0 the least-significant bit.
    plus = np.array([1, 1]) / math.sqrt(2)
    expected = np.kron(psi, np.kron(plus, plus))
    assert np.allclose(state, expected, atol=ATOL)


# --- 5. Dagger (uncompute) round-trips ---------------------------------------

def _random_circuit(n_qubits: int, depth: int, seed: int) -> Circuit:
    rng = random.Random(seed)
    c = Circuit(n_qubits)
    one_q = ['h', 'x', 'y', 'z', 's', 't', 'sx']
    rot = ['rx', 'ry', 'rz', 'phase']
    for _ in range(depth):
        kind = rng.random()
        q = rng.randrange(n_qubits)
        if kind < 0.35:
            getattr(c, rng.choice(one_q))[q]
        elif kind < 0.65:
            getattr(c, rng.choice(rot))(rng.uniform(0, 2 * math.pi))[q]
        else:
            q2 = rng.choice([i for i in range(n_qubits) if i != q])
            name = rng.choice(['cx', 'cz', 'swap', 'crz', 'cphase'])
            if name in ('crz', 'cphase'):
                getattr(c, name)(rng.uniform(0, 2 * math.pi))[q, q2]
            else:
                getattr(c, name)[q, q2]
    return c


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_dagger_uncomputes_random_circuit(seed):
    c = _random_circuit(4, 25, seed)
    state = (c + c.dagger()).run()
    expected = torch.zeros(16, dtype=torch.complex128)
    expected[0] = 1.0
    assert torch.allclose(state, expected, atol=1e-7)


def test_dagger_rejects_measurement():
    with pytest.raises(ValueError):
        Circuit(1).h[0].m[0].dagger()


def test_dagger_ignore_measurement_drops_it():
    c = Circuit(1).h[0].m[0].dagger(ignore_measurement=True)
    assert len(c.ops) == 1 and c.ops[0].lowername == 'h'


# --- 6. Cross-backend consistency --------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12])
def test_statevector_and_tensornet_agree(seed):
    c = _random_circuit(4, 30, seed)
    sv = c.run(backend="statevector")
    tn = c.run(backend="tensornet")
    assert torch.allclose(sv, tn, atol=1e-8)


# --- 7. Trotterized time evolution vs exact matrix exponential ----------------

@pytest.mark.parametrize("chars,coeff,t", [
    ("Z", 0.7, 0.9),
    ("X", -1.3, 0.4),
    ("Y", 0.5, 1.7),
    ("XZ", 0.8, 0.6),
    ("ZYX", -0.6, 1.1),
])
def test_time_evolution_matches_matrix_exponential(chars, coeff, t):
    # term_from_chars("XZ") = X0*Z1; a single Pauli term's evolution circuit is
    # exact (no Trotter error), so it must equal expm(-i t H) exactly.
    term = (coeff * term_from_chars(chars)).simplify()
    n = len(chars)
    h_mat = term.to_matrix(n).to(torch.complex128)
    expected_u = torch.matrix_exp(-1j * t * h_mat).numpy()

    c = Circuit(n)
    term.get_time_evolution()(c, t)
    assert _allclose_up_to_global_phase(_u(c), expected_u, atol=1e-7), \
        "time evolution circuit deviates from expm(-i t H)"


def test_time_evolution_rejects_complex_coefficient():
    with pytest.raises(ValueError):
        ((1 + 1j) * Z[0]).to_term().get_time_evolution()


def test_get_energy_y_observable_sign():
    # Regression: the sampler-based get_energy used to measure -Y instead of Y
    # (wrong direction of the RX basis rotation), flipping the sign of every
    # term with an odd number of Y operators. <Y> on RX(0.9)|0> is -sin(0.9).
    from blueqat.utils import AnsatzBase, non_sampling_sampler

    class OneParamAnsatz(AnsatzBase):
        def get_circuit(self, params):
            return Circuit(1).rx(params[0])[0]

    ansatz = OneParamAnsatz((1.0 * Y[0]).to_expr().simplify(), 1)
    ansatz.make_sparse(sparse=False)
    c = ansatz.get_circuit(torch.tensor([0.9], dtype=torch.float64))
    e_sampled = ansatz.get_energy(c, non_sampling_sampler).item()
    e_exact = ansatz.get_energy_sparse(c).item()
    assert e_sampled == pytest.approx(-math.sin(0.9), abs=1e-8)
    assert e_sampled == pytest.approx(e_exact, abs=1e-8)


# --- 8. Shot statistics sanity -------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_bell_shot_statistics(mode):
    torch.manual_seed(1234)
    shots = 10000
    counts = Circuit(2).h[0].cx[0, 1].shots(shots, backend=mode)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == shots
    # ~50/50 with generous 5-sigma bounds (sigma = sqrt(N)/2 = 50)
    assert abs(counts["00"] - shots / 2) < 5 * math.sqrt(shots) / 2 + 1


@pytest.mark.parametrize("mode", MODES)
def test_deterministic_shots(mode):
    counts = Circuit(2).x[0].shots(100, backend=mode)
    assert counts == {"01": 100}


# --- 9. Regression tests for audit fixes ---------------------------------------

def test_fixed_gates_reject_bogus_params():
    for build in [
        lambda: Circuit(1).x(0.5)[0],
        lambda: Circuit(1).h(0.5)[0],
        lambda: Circuit(2).cx(0.5)[0, 1],
        lambda: Circuit(2).zz(0.5)[0, 1],
        lambda: Circuit(2).swap(0.5)[0, 1],
        lambda: Circuit(3).ccx(0.5)[0, 1, 2],
    ]:
        with pytest.raises(ValueError):
            build()


def test_margolus_is_relative_phase_toffoli():
    import blueqat.macros  # noqa: F401  (registers the macro)
    u_marg = _u(Circuit(3).margolus(0, 1, 2))
    u_ccx = _u(Circuit(3).ccx[0, 1, 2])
    # Same absolute amplitudes as Toffoli (differs only in relative phases)...
    assert np.allclose(np.abs(u_marg), np.abs(u_ccx), atol=ATOL)
    # ...and still unitary.
    assert np.allclose(u_marg @ u_marg.conj().T, np.eye(8), atol=ATOL)


def test_json_roundtrip_three_qubit_gates():
    from blueqat.circuit_funcs.json_serializer import serialize, deserialize
    c = Circuit(3).h[0].ccx[0, 1, 2].cswap[0, 1, 2].ccz[0, 1, 2]
    c2 = deserialize(serialize(c))
    assert torch.allclose(c.run(), c2.run(), atol=ATOL)


def test_qasm_roundtrip_extended_gate_set():
    from blueqat.circuit_funcs.qasm_parser import from_qasm
    c = (Circuit(2).sx[0].sxdg[1].crx(0.5)[0, 1].cry(0.6)[0, 1].crz(0.7)[0, 1]
         .rxx(0.3)[0, 1].ryy(0.4)[0, 1].rzz(0.5)[0, 1].zz[0, 1])
    c2 = from_qasm(c.to_qasm())
    v1 = c.run().detach().numpy()
    v2 = c2.run().detach().numpy()
    # zz -> rzz(pi/2) drops a global phase, so compare via fidelity.
    assert abs(np.vdot(v1, v2)) == pytest.approx(1.0, abs=1e-8)
