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

    # QAOA is a stochastic heuristic, and even with a fixed seed the
    # optimization trajectory depends on platform floating-point details, so
    # a single run is not reproducible everywhere. Restart from a few seeds
    # (standard QAOA practice) and keep the best partition seen; each restart
    # inspects the top-10 most likely bitstrings.
    best_bits, best_diff = None, None
    for seed in (42, 0, 1, 2, 3):
        torch.manual_seed(seed)
        vqe = Vqe(QaoaAnsatz(hamiltonian, step=4))
        result = vqe.run(max_iter=500)
        for bits, _ in result.most_common(10):
            group0 = [x for x, b in zip(nums, bits) if b == 0]
            group1 = [x for x, b in zip(nums, bits) if b == 1]
            diff = abs(sum(group0) - sum(group1))
            if best_diff is None or diff < best_diff:
                best_bits, best_diff = bits, diff
        if best_diff == 1:
            break

    group0 = [x for x, b in zip(nums, best_bits) if b == 0]
    group1 = [x for x, b in zip(nums, best_bits) if b == 1]
    print(f"Best bitstring found: {best_bits}")
    print(f"Group 0 (sum={sum(group0)}): {group0}")
    print(f"Group 1 (sum={sum(group1)}): {group1}")
    print(f"Difference: {best_diff}")
    # The total (37) is odd, so a difference of 1 is the best possible.
    assert best_diff == 1, f"expected the optimal partition (diff 1), got {best_diff}"
