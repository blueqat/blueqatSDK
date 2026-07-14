"""Quantum Fourier Transform: the workhorse behind phase estimation and Shor.

The QFT maps |j> to (1/sqrt(N)) sum_k exp(2 pi i jk / N) |k> -- a discrete
Fourier transform of the amplitude vector, done in O(n^2) gates instead of the
classical O(N log N). This script builds the textbook H + controlled-phase
ladder, checks it against the DFT matrix, and uses it to read out the period
of a simple periodic state.
"""
import cmath
import math

import numpy as np

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary


def qft(n: int) -> Circuit:
    """Textbook QFT circuit on n qubits (qubit 0 = least-significant bit)."""
    c = Circuit(n)
    for i in reversed(range(n)):
        c.h[i]
        for k in range(i):
            c.cphase(math.pi / 2 ** (i - k))[k, i]
    for i in range(n // 2):          # bit-reversal swaps
        c.swap[i, n - 1 - i]
    return c


if __name__ == "__main__":
    n = 3
    dim = 2 ** n
    print(f"Quantum Fourier Transform on {n} qubits")
    print("=" * 50)

    # 1. The circuit's unitary must equal the DFT matrix.
    omega = cmath.exp(2j * math.pi / dim)
    dft = np.array([[omega ** (j * k) for j in range(dim)]
                    for k in range(dim)]) / math.sqrt(dim)
    u = circuit_to_unitary(qft(n))
    print(f"Matches DFT matrix: {np.allclose(u, dft, atol=1e-8)}")
    assert np.allclose(u, dft, atol=1e-8)

    # 2. QFT of a period-2 state (|0> + |2> + |4> + |6>)/2 concentrates all
    #    probability on multiples of N/period = 4, i.e. |0> and |4>.
    initial = np.zeros(dim, dtype=complex)
    initial[::2] = 0.5
    state = qft(n).run(initial=initial)
    probs = (state.abs() ** 2).numpy().round(6)
    print(f"QFT of period-2 state, P(k): {dict((k, p) for k, p in enumerate(probs) if p > 1e-9)}")
    assert abs(probs[0] - 0.5) < 1e-8 and abs(probs[4] - 0.5) < 1e-8

    # 3. QFT followed by its dagger is the identity.
    c = qft(n)
    roundtrip = (c + c.dagger()).run(initial=initial)
    assert np.allclose(roundtrip.numpy(), initial, atol=1e-8)
    print("OK: unitary check, period readout, and dagger round-trip all pass.")
