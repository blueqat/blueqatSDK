"""Number partitioning via QAOA.

Given a list of numbers, split them into two groups whose sums are as close
as possible. Encoding group membership as a spin (+-1) per number, the
imbalance squared (sum_i x_i * s_i) ** 2 is minimized exactly when the two
groups' sums match -- a natural QAOA cost Hamiltonian once s_i is replaced by
the Pauli operator Z_i.

QAOA is a heuristic: at low depth it usually doesn't put all its probability
on the single best answer, so in practice you sample several of the most
likely bitstrings and pick the best one, as this script does.
"""
from typing import List

import torch

from blueqat.utils import Vqe, QaoaAnsatz, Z

torch.manual_seed(42)


def numpartition_hamiltonian(nums: List[int]):
    """Cost Hamiltonian for partitioning `nums` into two equal-sum groups."""
    imbalance = 0
    for i, x in enumerate(nums):
        imbalance = imbalance + x * Z[i]
    return (imbalance * imbalance).simplify()


if __name__ == "__main__":
    print("Number partitioning via QAOA")
    print("=" * 50)

    nums = [3, 2, 6, 9, 2, 5, 7, 3]
    print(f"Numbers: {nums} (total: {sum(nums)})")

    hamiltonian = numpartition_hamiltonian(nums)
    vqe = Vqe(QaoaAnsatz(hamiltonian, step=3))
    result = vqe.run(max_iter=500)

    # Look at the top candidates and keep the best-balanced one.
    candidates = result.most_common(5)
    best_bits, best_diff = None, None
    for bits, _ in candidates:
        group0 = [x for x, b in zip(nums, bits) if b == 0]
        group1 = [x for x, b in zip(nums, bits) if b == 1]
        diff = abs(sum(group0) - sum(group1))
        if best_diff is None or diff < best_diff:
            best_bits, best_diff = bits, diff

    group0 = [x for x, b in zip(nums, best_bits) if b == 0]
    group1 = [x for x, b in zip(nums, best_bits) if b == 1]
    print(f"Best bitstring found: {best_bits}")
    print(f"Group 0 (sum={sum(group0)}): {group0}")
    print(f"Group 1 (sum={sum(group1)}): {group1}")
    print(f"Difference: {best_diff}")
