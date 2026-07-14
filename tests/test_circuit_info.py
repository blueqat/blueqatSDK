"""Tests for circuit introspection & convenience APIs added for parity with
other quantum SDKs: depth(), count_ops(), probs(), expect(), plus the iswap
gate and barrier operation."""
import math

import numpy as np
import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.utils import X, Z


# --- depth / count_ops --------------------------------------------------------

def test_depth_empty():
    assert Circuit(3).depth() == 0


def test_depth_parallel_gates_count_once():
    # h on all 3 qubits is depth 1, not 3.
    assert Circuit(3).h[:].depth() == 1


def test_depth_sequential_chain():
    c = Circuit(3).h[:].cx[0, 1].cx[1, 2].m[:]
    # h(1) -> cx01(2) -> cx12(3) -> m(4)
    assert c.depth() == 4


def test_depth_independent_qubits():
    c = Circuit(2).h[0].h[0].h[0].x[1]
    assert c.depth() == 3


def test_barrier_does_not_add_depth():
    assert Circuit(2).h[:].barrier[:].x[:].depth() == 2


def test_count_ops_expands_slices():
    c = Circuit(3).h[:].cx[0, 1].cx[1, 2].barrier[:].m[:]
    counts = c.count_ops()
    assert counts == {'h': 3, 'cx': 2, 'barrier': 1, 'measure': 3}


def test_count_ops_three_qubit_gate():
    assert Circuit(3).ccx[0, 1, 2].count_ops() == {'ccx': 1}


# --- probs --------------------------------------------------------------------

def test_probs_full():
    p = Circuit(2).h[0].cx[0, 1].probs()
    assert torch.allclose(p, torch.tensor([0.5, 0.0, 0.0, 0.5], dtype=p.dtype), atol=1e-8)


def test_probs_marginal_single_qubit():
    # qubit 0 in |1>, qubit 1 in |+>: marginal of qubit 1 is 50/50.
    c = Circuit(2).x[0].h[1]
    assert torch.allclose(c.probs([0]), torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-8)
    assert torch.allclose(c.probs([1]), torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-8)


def test_probs_qubit_order():
    # State |q1 q0> = |01>: probs([0, 1]) indexes bit0=q0, bit1=q1 -> P(index 1)=1;
    # probs([1, 0]) swaps the roles -> P(index 2)=1.
    c = Circuit(2).x[0]
    p01 = c.probs([0, 1])
    p10 = c.probs([1, 0])
    assert p01[1].item() == pytest.approx(1.0)
    assert p10[2].item() == pytest.approx(1.0)


def test_probs_sum_to_one_on_marginal():
    c = Circuit(3).h[:].cx[0, 1].crz(0.3)[1, 2]
    for qubits in ([0], [1, 2], [2, 0], None):
        p = c.probs(qubits)
        assert p.sum().item() == pytest.approx(1.0, abs=1e-8)


def test_probs_rejects_bad_qubits():
    with pytest.raises(ValueError):
        Circuit(2).h[0].probs([0, 0])
    with pytest.raises(ValueError):
        Circuit(2).h[0].probs([2])


def test_probs_is_differentiable():
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    p = Circuit(1).rx(theta)[0].probs([0])
    p[1].backward()
    # P(1) = sin^2(theta/2), dP/dtheta = sin(theta)/2
    assert theta.grad.item() == pytest.approx(math.sin(0.4) / 2, abs=1e-8)


# --- expect ---------------------------------------------------------------------

def test_expect_z_observable():
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    e = Circuit(1).rx(theta)[0].expect(1.0 * Z[0])
    assert e.item() == pytest.approx(math.cos(0.4), abs=1e-7)
    e.backward()
    assert theta.grad.item() == pytest.approx(-math.sin(0.4), abs=1e-7)


def test_expect_multi_term_hamiltonian():
    # H = 0.5*Z0 + 2.0*X1 on |0>|+> gives 0.5 + 2.0 = 2.5.
    c = Circuit(2).h[1]
    e = c.expect(0.5 * Z[0] + 2.0 * X[1])
    assert e.item() == pytest.approx(2.5, abs=1e-7)


def test_expect_accepts_bare_pauli():
    e = Circuit(1).expect(Z[0])
    assert e.item() == pytest.approx(1.0, abs=1e-8)


# --- iswap / barrier -------------------------------------------------------------

ISWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
])


def test_iswap_matrix():
    assert np.allclose(circuit_to_unitary(Circuit(2).iswap[0, 1]), ISWAP_MATRIX, atol=1e-8)


def test_iswap_fallback_matches_matrix():
    # The statevector backend runs iswap natively from matrix(); running the
    # fallback decomposition explicitly must give the same unitary.
    from blueqat.gate import ISwapGate
    fb = Circuit(2, ISwapGate((0, 1)).fallback(2))
    assert np.allclose(circuit_to_unitary(fb), ISWAP_MATRIX, atol=1e-8)


def test_iswap_dagger_roundtrip():
    u = circuit_to_unitary(Circuit(2).iswap[0, 1].iswapdg[0, 1])
    assert np.allclose(u, np.eye(4), atol=1e-8)


def test_iswap_backends_agree():
    sv = Circuit(2).h[0].iswap[0, 1].run(backend='statevector')
    tn = Circuit(2).h[0].iswap[0, 1].run(backend='tensornet')
    assert torch.allclose(sv, tn, atol=1e-8)


def test_barrier_is_identity_in_simulation():
    with_b = Circuit(2).h[0].barrier[:].cx[0, 1].run()
    without = Circuit(2).h[0].cx[0, 1].run()
    assert torch.allclose(with_b, without, atol=1e-12)


def test_barrier_in_qasm_output():
    qasm = Circuit(3).h[0].barrier[:].to_qasm()
    assert "barrier q[0],q[1],q[2];" in qasm


def test_barrier_dagger_and_flatten():
    from blueqat.circuit_funcs.flatten import flatten
    c = Circuit(2).h[0].barrier[:].x[1]
    d = c.dagger()
    assert [op.lowername for op in d.ops] == ['x', 'barrier', 'h']
    f = flatten(c)
    assert [op.lowername for op in f.ops] == ['h', 'barrier', 'x']
    assert f.ops[1].targets == (0, 1)


def test_barrier_json_roundtrip():
    from blueqat.circuit_funcs.json_serializer import serialize, deserialize
    c = Circuit(2).h[0].barrier[:].cx[0, 1]
    c2 = deserialize(serialize(c))
    assert [op.lowername for op in c2.ops] == ['h', 'barrier', 'cx']
    assert torch.allclose(c.run(), c2.run(), atol=1e-12)
