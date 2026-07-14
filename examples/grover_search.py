"""Grover's search: find a marked item in an unsorted database of 8 entries.

A classical search over N unsorted items needs O(N) queries; Grover's algorithm
needs only O(sqrt(N)). Each iteration applies the "oracle" (which flips the
phase of the marked state) and the "diffusion" operator (inversion about the
mean), rotating the state a little closer to the marked item.

For N = 8 items (3 qubits), the optimal number of iterations is
round(pi/4 * sqrt(8)) = 2, giving a success probability of about 94.5%.
"""
import math
from collections import Counter

from blueqat import Circuit
import blueqat.macros  # noqa: F401  -- registers the mcz_gray macro


N_QUBITS = 3
MARKED = "101"  # the bitstring we want to find (qubit 0 is the rightmost bit)


def oracle(c: Circuit, marked: str) -> Circuit:
    """Flip the sign of |marked>. X-conjugation turns the all-ones controlled-Z
    into a phase flip on exactly the marked bitstring."""
    zeros = [i for i, b in enumerate(reversed(marked)) if b == "0"]
    if zeros:
        c.x[tuple(zeros)]
    c.mcz_gray(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    if zeros:
        c.x[tuple(zeros)]
    return c


def diffusion(c: Circuit) -> Circuit:
    """Inversion about the mean: H X (controlled-Z on all) X H."""
    c.h[:].x[:]
    c.mcz_gray(list(range(N_QUBITS - 1)), N_QUBITS - 1)
    c.x[:].h[:]
    return c


if __name__ == "__main__":
    print(f"Grover search for |{MARKED}> among {2**N_QUBITS} items")
    print("=" * 50)

    n_iterations = round(math.pi / 4 * math.sqrt(2 ** N_QUBITS))
    c = Circuit(N_QUBITS).h[:]          # uniform superposition
    for _ in range(n_iterations):
        oracle(c, MARKED)
        diffusion(c)

    probs = (c.run().abs() ** 2)
    p_marked = probs[int(MARKED, 2)].item()
    print(f"Iterations: {n_iterations}")
    print(f"P(|{MARKED}>) = {p_marked:.4f}  (theory: {math.sin((2*n_iterations+1)*math.asin(1/math.sqrt(8)))**2:.4f})")

    counts: Counter = c.m[:].shots(1000)
    print(f"1000 shots: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    assert p_marked > 0.9, "Grover should amplify the marked state above 90%"
    assert counts.most_common(1)[0][0] == MARKED
    print("OK: the marked item was found with high probability.")
