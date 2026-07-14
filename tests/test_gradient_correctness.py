"""Gradient correctness tests for the PyTorch backend, in both mode=statevector
and mode=tensornet.

Earlier bugs in this codebase silently detached the autograd graph (e.g. a
`torch.tensor([...])` built from tensor elements instead of `torch.stack`) while
still producing numerically plausible-looking output, so a test that only checks
the loss value (or a loss that happens to be tautologically 1.0 due to state
normalization) would not have caught them. These tests instead check the actual
`.grad` against a closed-form analytic derivative.
"""
import pytest
import torch

from blueqat import Circuit

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

MODES = ["statevector", "tensornet"]


def _kron_obs(op: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Embed a 1-qubit observable into an n_qubits operator, qubit 0 = LSB
    (rightmost factor), matching the statevector index convention used
    throughout this codebase (index = sum bit_i * 2**i)."""
    I2 = torch.eye(2, dtype=torch.complex128)
    m = op if (n_qubits - 1) == qubit else I2
    for q in range(n_qubits - 2, -1, -1):
        m = torch.kron(m, op if q == qubit else I2)
    return m


def _expval(state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.real((state.conj() * (obs @ state)).sum())


@pytest.mark.parametrize("mode", MODES)
def test_rx_gradient_matches_analytic(mode):
    # RX(theta)|0>: <Z> = cos(theta), d<Z>/dtheta = -sin(theta).
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    state = Circuit(1).rx(theta)[0].run(backend=mode)
    loss = _expval(state, Z)

    assert not torch.allclose(loss.detach(), torch.tensor(1.0, dtype=torch.float64)), \
        "loss is tautologically 1.0 -- not a meaningful gradient test"
    assert torch.allclose(loss.detach(), torch.cos(theta.detach()), atol=1e-6)

    loss.backward()
    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_ry_gradient_matches_analytic(mode):
    # RY(theta)|0>: <Z> = cos(theta), d<Z>/dtheta = -sin(theta).
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    state = Circuit(1).ry(theta)[0].run(backend=mode)
    loss = _expval(state, Z)

    assert torch.allclose(loss.detach(), torch.cos(theta.detach()), atol=1e-6)
    loss.backward()
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_rz_gradient_matches_analytic(mode):
    # RZ(theta) H|0>: <X> = cos(theta), d<X>/dtheta = -sin(theta).
    theta = torch.tensor(0.55, dtype=torch.float64, requires_grad=True)
    state = Circuit(1).h[0].rz(theta)[0].run(backend=mode)
    loss = _expval(state, X)

    assert torch.allclose(loss.detach(), torch.cos(theta.detach()), atol=1e-6)
    loss.backward()
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_crz_gradient_matches_analytic_when_control_active(mode):
    # Direct regression test for the crz gradient-detachment bug: with the
    # control qubit forced to |1>, CRZ(theta)[0, 1] reduces to a plain
    # RZ(theta) on the target qubit, so <X_target> = cos(theta) exactly.
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    state = Circuit(2).x[0].h[1].crz(theta)[0, 1].run(backend=mode)
    loss = _expval(state, _kron_obs(X, 1, 2))

    assert not torch.allclose(loss.detach(), torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(loss.detach(), torch.cos(theta.detach()), atol=1e-6)

    loss.backward()
    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_crz_gradient_is_zero_when_control_inactive(mode):
    # Contrast case: with the control qubit left at |0>, CRZ has no effect on
    # the target at all, so the gradient must be exactly zero. This confirms
    # the previous test is actually exercising CRZ's theta-dependence and not
    # passing vacuously.
    theta = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    state = Circuit(2).h[1].crz(theta)[0, 1].run(backend=mode)
    loss = _expval(state, _kron_obs(X, 1, 2))

    loss.backward()
    assert theta.grad is not None
    assert torch.allclose(theta.grad, torch.tensor(0.0, dtype=torch.float64), atol=1e-6)
