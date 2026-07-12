"""Blueqat basics: build a Bell state, inspect its statevector, and sample it.

A Bell state (|00> + |11>) / sqrt(2) is the simplest example of entanglement:
measuring either qubit is a fair coin flip, but the two outcomes are always
identical. This script builds one with a Hadamard and a CNOT, and shows the
statevector, single-amplitude, and shot-sampling ways to inspect a circuit.
"""
from collections import Counter

from blueqat import Circuit


if __name__ == "__main__":
    print("Bell state basics")
    print("=" * 50)

    circuit = Circuit(2).h[0].cx[0, 1]

    # 1. The full statevector (a torch.Tensor, gradient-friendly).
    state = circuit.run()
    print(f"Statevector: {state}")

    # 2. A single amplitude, without ever building the full vector.
    amp_00 = circuit.run(returns="amplitude", amplitude="00")
    print(f"Amplitude of |00>: {amp_00}")

    # 3. Sampling: measuring collapses the superposition into shot outcomes.
    counts: Counter = circuit.m[:].run(shots=1000)
    print(f"1000 shots: {counts}")
    assert set(counts) <= {"00", "11"}, "Bell state should only ever show 00 or 11"
    print("(only '00' and '11' ever appear -- that's the entanglement)")
