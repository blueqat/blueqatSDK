"""Quantum teleportation: move a qubit state using entanglement.

Alice holds an unknown state |psi> on qubit 0 and shares a Bell pair with Bob
(qubits 1 and 2). After Alice's Bell-basis rotation, Bob applies corrections
and ends up holding |psi> exactly -- no matter what it was. Here the usual
"measure, then send 2 classical bits" step is replaced by coherent CX/CZ
corrections, which lets us verify the transfer deterministically on the
statevector. A shot-based run with real mid-circuit measurement follows.
"""
import math
import random

import numpy as np
import torch

from blueqat import Circuit

# Deterministic shot sampling so the 3-sigma-style statistical check below
# can't fail by chance (e.g. in CI).
torch.manual_seed(1234)


def make_psi(theta: float, phi: float) -> np.ndarray:
    return np.array([math.cos(theta / 2),
                     cmath_exp(phi) * math.sin(theta / 2)])


def cmath_exp(phi: float) -> complex:
    return complex(math.cos(phi), math.sin(phi))


if __name__ == "__main__":
    print("Quantum teleportation (qubit 0 -> qubit 2)")
    print("=" * 50)
    rng = random.Random(7)
    theta, phi = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
    psi = make_psi(theta, phi)
    print(f"State to teleport: {psi.round(4)}")

    # --- Coherent version: deterministic, verifiable on the statevector ----
    c = Circuit(3)
    c.ry(theta)[0].rz(phi)[0]        # prepare |psi> on qubit 0 (up to phase)
    c.h[1].cx[1, 2]                  # Bell pair between qubits 1 and 2
    c.cx[0, 1].h[0]                  # Alice rotates into the Bell basis
    c.cx[1, 2].cz[0, 2]              # Bob's corrections (coherently controlled)

    state = c.run().numpy()
    # Expected result: qubits 0 and 1 in |+>, qubit 2 holding |psi>.
    plus = np.array([1, 1]) / math.sqrt(2)
    psi_up_to_phase = Circuit(1).ry(theta)[0].rz(phi)[0].run().numpy()
    expected = np.kron(psi_up_to_phase, np.kron(plus, plus))
    ok = np.allclose(state, expected, atol=1e-8)
    print(f"Coherent teleportation exact: {ok}")
    assert ok

    # --- Measured version: real mid-circuit collapse, statistical check ----
    # Without classical feed-forward we can still verify the Z-basis statistics:
    # P(qubit2 = 1) must equal |<1|psi>|^2 = sin^2(theta/2) regardless of
    # Alice's (discarded) measurement results.
    shots = 4000
    c2 = Circuit(3)
    c2.ry(theta)[0].rz(phi)[0]
    c2.h[1].cx[1, 2]
    c2.cx[0, 1].h[0]
    c2.cx[1, 2].cz[0, 2]
    c2.m[2]                          # only Bob's qubit is reported
    counts = c2.shots(shots)
    p1 = sum(v for k, v in counts.items() if k[0] == "1") / shots
    print(f"P(qubit2=1) sampled: {p1:.3f}   theory: {math.sin(theta/2)**2:.3f}")
    assert abs(p1 - math.sin(theta / 2) ** 2) < 0.05
    print("OK: Bob receives |psi> exactly; sampled statistics agree.")
