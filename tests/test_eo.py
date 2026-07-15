"""Tests for exchange-only (EO) spin-qubit support: the exch gate primitive,
the 3-spin DFS encoding, analytic pulse sequences (including the serial
Fong-Wandzura CNOT), the 'eo' transpiler backend, and differentiable gate
synthesis."""
import math

import numpy as np
import pytest
import torch

import blueqat.eo  # noqa: F401  (registers the 'eo' backend)
from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.eo import encoding, synthesize_1q
from blueqat.eo.sequences import (cx_sequence, cz_sequence, h_sequence,
                                  rx_sequence, ry_sequence, rz_sequence,
                                  sequence_to_circuit, swap_sequence,
                                  x_sequence, y_sequence, z_sequence)

ATOL = 1e-8

X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)


def _u(seq, n):
    return torch.tensor(np.array(circuit_to_unitary(sequence_to_circuit(seq, n))),
                        dtype=torch.complex128)


def _rz(phi):
    return torch.tensor([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
                        dtype=torch.complex128)


# --- exch gate primitive -----------------------------------------------------

def test_exch_pi_is_exact_swap():
    m = Circuit(2).exch(math.pi)[0, 1]
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    assert np.allclose(circuit_to_unitary(m), swap, atol=ATOL)


def test_exch_unitary_and_dagger():
    from blueqat.gate import ExchangeGate
    g = ExchangeGate((0, 1), 0.37)
    m = g.matrix()
    assert torch.allclose(m @ m.conj().mT, torch.eye(4, dtype=torch.complex128), atol=1e-10)
    assert torch.allclose(g.dagger().matrix(), m.conj().mT, atol=1e-10)


def test_exch_phases_singlet_only():
    theta = 0.9
    m = Circuit(2).exch(theta)[0, 1].run(initial=torch.tensor(
        [0, 1, -1, 0], dtype=torch.complex128) / math.sqrt(2))
    expected = torch.tensor([0, 1, -1, 0], dtype=torch.complex128) / math.sqrt(2)
    expected = expected * complex(np.exp(1j * theta))
    assert torch.allclose(m, expected, atol=ATOL)


def test_exch_backends_agree():
    sv = Circuit(2).h[0].exch(0.7)[0, 1].run(backend='statevector')
    tn = Circuit(2).h[0].exch(0.7)[0, 1].run(backend='tensornet')
    assert torch.allclose(sv, tn, atol=ATOL)


def test_exch_fallback_matches_matrix_up_to_phase():
    from blueqat.gate import ExchangeGate
    g = ExchangeGate((0, 1), 0.61)
    fb = Circuit(2, g.fallback(2))
    u_fb = torch.tensor(np.array(circuit_to_unitary(fb)), dtype=torch.complex128)
    fid = encoding.logical_fidelity(u_fb, g.matrix())
    assert fid == pytest.approx(1.0, abs=1e-10)


def test_exch_gradient_flows():
    theta = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    state = Circuit(2).x[0].exch(theta)[0, 1].run()
    # amplitude of |10> after partial swap: (1 - e^{i theta})/2
    p10 = (state[2].abs() ** 2)
    p10.backward()
    # |b|^2 = sin^2(theta/2), d/dtheta = sin(theta)/2
    assert theta.grad.item() == pytest.approx(math.sin(0.5) / 2, abs=1e-8)


# --- encoding ------------------------------------------------------------------

def test_codewords_orthonormal():
    for m in ['+', '-']:
        b = encoding.codeword_basis(m)
        gram = b.conj().T @ b
        assert torch.allclose(gram, torch.eye(2, dtype=torch.complex128), atol=1e-12)


def test_codewords_have_zero_leakage():
    for m in ['+', '-']:
        b = encoding.codeword_basis(m)
        for k in range(2):
            assert encoding.leakage(b[:, k]) < 1e-24


def test_fully_polarized_state_is_pure_leakage():
    up3 = torch.zeros(8, dtype=torch.complex128)
    up3[0] = 1.0
    assert encoding.leakage(up3) == pytest.approx(1.0)


def test_encode_state_roundtrip():
    state = encoding.encode_state([(0.6, 0.8)])
    b = encoding.codeword_basis('+')
    amps = b.conj().T @ state
    assert amps[0].real == pytest.approx(0.6, abs=1e-12)
    assert amps[1].real == pytest.approx(0.8, abs=1e-12)


# --- analytic sequences ----------------------------------------------------------

@pytest.mark.parametrize("m", ['+', '-'])
@pytest.mark.parametrize("seq_fn,target,name", [
    (x_sequence, X, 'X'),
    (y_sequence, Y, 'Y'),
    (z_sequence, Z, 'Z'),
    (h_sequence, H, 'H'),
])
def test_fixed_1q_sequences(seq_fn, target, name, m):
    L = encoding.logical_action(_u(seq_fn(), 3), m)
    assert encoding.logical_fidelity(L, target) == pytest.approx(1.0, abs=1e-9), name


@pytest.mark.parametrize("m", ['+', '-'])
@pytest.mark.parametrize("phi", [0.3, math.pi / 2, -1.2, 2 * math.pi - 0.01])
def test_rz_sequence(phi, m):
    L = encoding.logical_action(_u(rz_sequence(phi), 3), m)
    assert encoding.logical_fidelity(L, _rz(phi)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("phi", [0.4, -1.1])
def test_rx_ry_sequences(phi):
    rx_t = torch.tensor([[math.cos(phi / 2), -1j * math.sin(phi / 2)],
                         [-1j * math.sin(phi / 2), math.cos(phi / 2)]],
                        dtype=torch.complex128)
    ry_t = torch.tensor([[math.cos(phi / 2), -math.sin(phi / 2)],
                         [math.sin(phi / 2), math.cos(phi / 2)]],
                        dtype=torch.complex128)
    L = encoding.logical_action(_u(rx_sequence(phi), 3), '+')
    assert encoding.logical_fidelity(L, rx_t) == pytest.approx(1.0, abs=1e-9)
    L = encoding.logical_action(_u(ry_sequence(phi), 3), '+')
    assert encoding.logical_fidelity(L, ry_t) == pytest.approx(1.0, abs=1e-9)


def test_1q_sequences_gauge_sector_actions_identical():
    # Not just fidelity: the two gauge sectors' logical blocks must match
    # including phase, otherwise gauge superpositions would decohere.
    for seq in [x_sequence(), h_sequence(), rz_sequence(0.7)]:
        u = _u(seq, 3)
        lp = encoding.logical_action(u, '+')
        lm = encoding.logical_action(u, '-')
        assert torch.allclose(lp, lm, atol=1e-12)


# --- Fong-Wandzura CNOT -----------------------------------------------------------

CX_TARGET = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=torch.complex128)  # control = logical qubit 1, target = logical qubit 0


def test_fw_cnot_pulse_count():
    assert len(cx_sequence(3, 0)) == 28


def test_fw_cnot_all_gauge_sectors():
    u = _u(cx_sequence(3, 0), 6)
    blocks = {}
    for m1 in ['+', '-']:
        for m2 in ['+', '-']:
            blocks[(m1, m2)] = encoding.two_qubit_logical_action(u, m1, m2)
            fid = encoding.logical_fidelity(blocks[(m1, m2)], CX_TARGET)
            assert fid == pytest.approx(1.0, abs=1e-9), (m1, m2)
    # Truly gauge-independent: identical blocks including phase.
    ref = blocks[('+', '+')]
    for b in blocks.values():
        assert torch.allclose(b, ref, atol=1e-12)


def test_encoded_cz_is_symmetric_diag():
    u = _u(cz_sequence(3, 0), 6)
    L = encoding.two_qubit_logical_action(u, '+', '+')
    cz = torch.diag(torch.tensor([1, 1, 1, -1], dtype=torch.complex128))
    assert encoding.logical_fidelity(L, cz) == pytest.approx(1.0, abs=1e-9)


def test_encoded_swap():
    u = _u(swap_sequence(0, 3), 6)
    L = encoding.two_qubit_logical_action(u, '+', '+')
    swap = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=torch.complex128)
    assert encoding.logical_fidelity(L, swap) == pytest.approx(1.0, abs=1e-9)


# --- 'eo' transpiler backend --------------------------------------------------------

def test_eo_backend_bell_state_end_to_end():
    phys = Circuit(2).h[0].cx[0, 1].run(backend='eo')
    assert phys.n_qubits == 6
    assert all(op.lowername == 'exch' for op in phys.ops)
    assert len(phys.ops) == 3 + 28

    init = encoding.encode_state([(1, 0), (1, 0)])
    final = phys.run(backend='statevector', initial=init)
    basis = encoding.two_qubit_codeword_basis('+', '+')
    amps = basis.conj().T.to(final.dtype) @ final
    bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex128) / math.sqrt(2)
    assert (torch.vdot(bell, amps).abs() ** 2).item() == pytest.approx(1.0, abs=1e-9)
    assert encoding.leakage(final, 0) < 1e-18
    assert encoding.leakage(final, 1) < 1e-18


def test_eo_backend_1q_gates_roundtrip():
    # A logical circuit whose net effect is identity must return the codeword.
    phys = Circuit(1).h[0].s[0].sdg[0].h[0].run(backend='eo')
    init = encoding.encode_state([(1, 0)])
    final = phys.run(initial=init)
    fid = (torch.vdot(init.to(final.dtype), final).abs() ** 2).item()
    assert fid == pytest.approx(1.0, abs=1e-9)


def test_eo_backend_rejects_unsupported():
    with pytest.raises(ValueError, match='not supported'):
        Circuit(3).ccx[0, 1, 2].run(backend='eo')


def test_eo_backend_run_with_alias():
    phys = Circuit(1).x[0].run_with_eo()
    assert phys.n_qubits == 3 and len(phys.ops) == 3


# --- differentiable synthesis ----------------------------------------------------------

def test_synthesize_arbitrary_su2_in_4_pulses():
    g = torch.Generator().manual_seed(7)
    a = torch.randn(2, 2, generator=g, dtype=torch.float64) \
        + 1j * torch.randn(2, 2, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    target = q * (torch.diagonal(r) / torch.diagonal(r).abs())

    seq = synthesize_1q(target, n_pulses=4, seed=42)
    assert len(seq) == 4
    u = _u(seq, 3)
    for m in ['+', '-']:
        L = encoding.logical_action(u, m)
        assert encoding.logical_fidelity(L, target) >= 1.0 - 1e-8


def test_synthesize_t_gate_shorter_than_composed_table():
    t_target = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]],
                            dtype=torch.complex128)
    seq = synthesize_1q(t_target, n_pulses=4, seed=42)
    L = encoding.logical_action(_u(seq, 3), '+')
    assert encoding.logical_fidelity(L, t_target) >= 1.0 - 1e-8


# --- 2q refinement, quantization, schedules (Phase 4/5) ---------------------------

def test_synthesize_2q_recalibrates_perturbed_fw_angles():
    # The calibration use case: perturb every FW-CNOT pulse area (as hardware
    # drift would), then let the differentiable optimizer pull the sequence
    # back to an exact, gauge-independent CNOT.
    from blueqat.eo import synthesize_2q
    fw = cx_sequence(3, 0)
    pairs = [p for p, _ in fw]
    g = torch.Generator().manual_seed(5)
    perturbed = [t + 0.05 * float(torch.randn((), generator=g)) for _, t in fw]

    u_pert = _u(list(zip(pairs, perturbed)), 6)
    basis = encoding.two_qubit_codeword_basis('+', '+')
    fid_pert = encoding.logical_fidelity(basis.conj().T @ u_pert @ basis, CX_TARGET)
    assert fid_pert < 0.999  # the perturbation is actually harmful

    refined = synthesize_2q(CX_TARGET, pairs=pairs, initial_thetas=perturbed,
                            n_restarts=1, seed=None)
    u_ref = _u(refined, 6)
    for m1 in ['+', '-']:
        for m2 in ['+', '-']:
            L = encoding.two_qubit_logical_action(u_ref, m1, m2)
            assert encoding.logical_fidelity(L, CX_TARGET) >= 1.0 - 1e-8


def test_quantize_sequence_fine_grid_keeps_fidelity():
    from blueqat.eo import quantize_sequence
    xq = quantize_sequence(x_sequence(), 2 * math.pi / 4096)
    L = encoding.logical_action(_u(xq, 3), '+')
    assert encoding.logical_fidelity(L, X) > 1.0 - 1e-4


def test_quantize_sequence_snaps_and_drops():
    from blueqat.eo import quantize_sequence
    step = math.pi / 2
    seq = [((0, 1), math.pi / 2 + 0.01), ((1, 2), 1e-9), ((0, 1), math.pi)]
    q = quantize_sequence(seq, step)
    assert q == [((0, 1), math.pi / 2), ((0, 1), math.pi)]
    with pytest.raises(ValueError):
        quantize_sequence(seq, 0.0)


def test_schedule_roundtrip_preserves_state():
    from blueqat.eo import from_schedule, to_schedule
    phys = Circuit(2).h[0].cx[0, 1].run(backend='eo')
    sched = to_schedule(phys)
    init = encoding.encode_state([(1, 0), (1, 0)])
    v1 = phys.run(initial=init)
    v2 = from_schedule(sched).run(initial=init)
    assert torch.allclose(v1, v2, atol=1e-10)


def test_schedule_is_json_serializable_and_versioned():
    import json
    from blueqat.eo import to_schedule
    sched = to_schedule(Circuit(1).h[0].run(backend='eo'))
    text = json.dumps(sched)
    assert '"blueqat-eo-schedule"' in text
    assert sched["version"] == "1"
    assert sched["n_spins"] == 3


def test_schedule_packs_disjoint_pulses_in_parallel():
    from blueqat.eo import schedule_stats, to_schedule
    # Pulses on (0,1) and (2,3) are disjoint -> run simultaneously.
    seq = [((0, 1), math.pi), ((2, 3), math.pi), ((1, 2), math.pi)]
    sched = to_schedule(seq)
    starts = {tuple(p["pair"]): p["start"] for p in sched["pulses"]}
    assert starts[(0, 1)] == starts[(2, 3)] == 0.0
    assert starts[(1, 2)] == pytest.approx(math.pi)
    stats = schedule_stats(sched)
    assert stats["parallel_speedup"] > 1.0
    assert stats["scheduled_duration"] == pytest.approx(2 * math.pi)


def test_schedule_never_overlaps_shared_spins():
    from blueqat.eo import to_schedule
    phys = Circuit(2).h[0].cx[0, 1].run(backend='eo')
    sched = to_schedule(phys)
    busy = {}
    for p in sched["pulses"]:
        for spin in p["pair"]:
            for (s, e) in busy.get(spin, []):
                assert p["start"] >= e - 1e-12 or p["start"] + p["duration"] <= s + 1e-12
            busy.setdefault(spin, []).append((p["start"], p["start"] + p["duration"]))


def test_schedule_canonicalizes_negative_and_periodic_theta():
    from blueqat.eo import from_schedule, to_schedule
    # A daggered exchange circuit has negative pulse areas; the exchange
    # unitary is exactly 2*pi-periodic, so the schedule must contain the
    # equivalent positive-duration pulses (never negative durations), and
    # exact no-ops (multiples of 2*pi) must be dropped.
    seq = [((0, 1), -0.7), ((0, 1), 7.0), ((1, 2), 0.0), ((1, 2), 2 * math.pi)]
    sched = to_schedule(seq)
    assert all(p["duration"] > 0 for p in sched["pulses"])
    assert len(sched["pulses"]) == 2  # the two no-ops are dropped
    assert sched["pulses"][0]["theta"] == pytest.approx(2 * math.pi - 0.7)
    assert sched["pulses"][1]["theta"] == pytest.approx(7.0 - 2 * math.pi)

    # The canonicalized schedule reproduces the original unitary.
    c1 = Circuit(3).exch(-0.7)[0, 1].exch(7.0)[0, 1]
    v1 = c1.run()
    v2 = from_schedule(to_schedule(c1)).run()
    assert torch.allclose(v1, v2, atol=1e-10)


def test_schedule_of_daggered_circuit():
    from blueqat.eo import from_schedule, to_schedule
    phys = Circuit(1).h[0].run(backend='eo')
    inv = phys.dagger()
    sched = to_schedule(inv)
    assert all(p["duration"] > 0 for p in sched["pulses"])
    init = encoding.encode_state([(1, 0)])
    v1 = inv.run(initial=init)
    v2 = from_schedule(sched).run(initial=init)
    assert torch.allclose(v1, v2, atol=1e-10)


def test_schedule_rejects_non_exchange_circuit():
    from blueqat.eo import to_schedule
    with pytest.raises(ValueError, match='exchange'):
        to_schedule(Circuit(2).h[0])


def test_exchange_circuit_serializes_for_cloud():
    # The 'exch' pulses must survive the JSON wire format used by the cloud
    # backend (schema round-trip through the ordinary serializer).
    from blueqat.circuit_funcs.json_serializer import deserialize, serialize
    phys = Circuit(1).h[0].run(backend='eo')
    c2 = deserialize(serialize(phys))
    init = encoding.encode_state([(1, 0)])
    assert torch.allclose(phys.run(initial=init), c2.run(initial=init), atol=1e-12)
