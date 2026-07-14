"""Unitarity and dagger/fallback consistency checks for every concrete gate in blueqat.gate.

This suite would have caught, for example, the CYGate.matrix() bug (a non-unitary,
det=0 matrix that was shipped for a while) and the ZZGate.dagger() bug (returning a
non-Hermitian-adjoint dagger). It exists to prevent that class of regression.
"""
import cmath

import pytest
import torch

from blueqat import Circuit, gate
from blueqat.backends.torch_backend import TorchBackend

ATOL = 1e-10

# (gate instance, n_qargs) for every concrete gate class with a real matrix().
# Mat1Gate is excluded: it is a pass-through wrapper for a user-supplied matrix,
# unitary only if the caller's input is, so it is exercised separately below.
UNITARY_GATE_CASES = [
    (gate.HGate((0,)), 1),
    (gate.IGate((0,)), 1),
    (gate.PhaseGate((0,), 0.37), 1),
    (gate.RXGate((0,), 0.37), 1),
    (gate.RYGate((0,), 0.53), 1),
    (gate.RZGate((0,), 0.71), 1),
    (gate.SGate((0,)), 1),
    (gate.SDagGate((0,)), 1),
    (gate.SXGate((0,)), 1),
    (gate.SXDagGate((0,)), 1),
    (gate.TGate((0,)), 1),
    (gate.TDagGate((0,)), 1),
    (gate.UGate((0,), 0.3, 0.5, 0.7, 0.2), 1),
    (gate.XGate((0,)), 1),
    (gate.YGate((0,)), 1),
    (gate.ZGate((0,)), 1),
    (gate.CHGate((0, 1)), 2),
    (gate.CPhaseGate((0, 1), 0.37), 2),
    (gate.CRXGate((0, 1), 0.37), 2),
    (gate.CRYGate((0, 1), 0.53), 2),
    (gate.CRZGate((0, 1), 0.71), 2),
    (gate.CUGate((0, 1), 0.3, 0.5, 0.7, 0.2), 2),
    (gate.CXGate((0, 1)), 2),
    (gate.CYGate((0, 1)), 2),
    (gate.CZGate((0, 1)), 2),
    (gate.RXXGate((0, 1), 0.37), 2),
    (gate.RYYGate((0, 1), 0.53), 2),
    (gate.RZZGate((0, 1), 0.71), 2),
    (gate.SwapGate((0, 1)), 2),
    (gate.ZZGate((0, 1)), 2),
    (gate.ZZDagGate((0, 1)), 2),
    (gate.ToffoliGate((0, 1, 2)), 3),
    (gate.CCZGate((0, 1, 2)), 3),
    (gate.CSwapGate((0, 1, 2)), 3),
]


def _case_id(case):
    g, _ = case
    return g.lowername


@pytest.mark.parametrize("case", UNITARY_GATE_CASES, ids=_case_id)
def test_matrix_is_unitary(case):
    g, n_qargs = case
    m = g.matrix()
    dim = 2 ** n_qargs
    assert m.shape == (dim, dim)
    identity = torch.eye(dim, dtype=torch.complex128)
    assert torch.allclose(m @ m.conj().mT, identity, atol=ATOL), \
        f"{g.lowername} matrix is not unitary: M @ M^dagger != I"


@pytest.mark.parametrize("case", UNITARY_GATE_CASES, ids=_case_id)
def test_dagger_matches_conjugate_transpose(case):
    g, _ = case
    m = g.matrix()
    dagger_m = g.dagger().matrix()
    assert torch.allclose(dagger_m, m.conj().mT, atol=ATOL), \
        f"{g.lowername}.dagger().matrix() is not the conjugate transpose of matrix()"


def test_mat1gate_arbitrary_unitary_roundtrip():
    # Hadamard as a stand-in "arbitrary" unitary.
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / cmath.sqrt(2)
    g = gate.Mat1Gate((0,), h)
    m = g.matrix()
    assert torch.allclose(m @ m.conj().mT, torch.eye(2, dtype=torch.complex128), atol=ATOL)
    assert torch.allclose(g.dagger().matrix(), m.conj().mT, atol=ATOL)


# --- fallback() consistency -------------------------------------------------
# For single-qubit IFallbackOperation gates, fallback() decomposes into
# PhaseGate calls on the same qubit, so the sub-matrices can be composed
# directly without any qubit embedding.
SINGLE_QUBIT_FALLBACK_CASES = [
    gate.SGate((0,)),
    gate.SDagGate((0,)),
    gate.TGate((0,)),
    gate.TDagGate((0,)),
]


@pytest.mark.parametrize("g", SINGLE_QUBIT_FALLBACK_CASES, ids=lambda g: g.lowername)
def test_single_qubit_fallback_matches_matrix(g):
    sub_gates = g.fallback(1)
    composed = torch.eye(2, dtype=torch.complex128)
    for sub in sub_gates:
        composed = sub.matrix() @ composed
    assert torch.allclose(composed, g.matrix(), atol=ATOL), \
        f"{g.lowername}.fallback() does not compose to the same unitary as matrix()"


def test_igate_fallback_is_empty():
    assert gate.IGate((0,)).fallback(1) == []


# For the 3-qubit fallback-only gates (Toffoli, CCZ, CSwap), the decomposition
# spans multiple qubits, so instead of hand-rolling matrix embedding, run the
# actual gate through Circuit + the statevector backend (which dispatches
# unrecognized names to `.fallback()`) and compare against matrix() applied
# directly to the same initial statevector.
THREE_QUBIT_FALLBACK_CASES = [
    gate.ToffoliGate((0, 1, 2)),
    gate.CCZGate((0, 1, 2)),
    gate.CSwapGate((0, 1, 2)),
]


@pytest.mark.parametrize("g", THREE_QUBIT_FALLBACK_CASES, ids=lambda g: g.lowername)
def test_three_qubit_fallback_matches_matrix(g):
    torch.manual_seed(0)
    backend = TorchBackend(mode="statevector")
    m = g.matrix()
    for _ in range(5):
        state = torch.randn(8, dtype=torch.complex128)
        state = state / state.norm()
        expected = m @ state
        actual = backend.run([g], 3, initial=state, returns="statevector")
        assert torch.allclose(actual, expected, atol=1e-8), \
            f"{g.lowername}: backend's fallback() execution diverges from matrix()"
