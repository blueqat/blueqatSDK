"""Max-Cut optimization via QAOA.

Given a graph, Max-Cut asks for a 2-coloring of its vertices that maximizes
the number of edges crossing between the two colors. Encoding "vertex i and j
have different colors" as minimizing <Z_i Z_j> gives a natural QAOA cost
Hamiltonian: H = sum over edges (i, j) of Z_i * Z_j.
"""
from typing import Callable, List, Optional, Tuple

from blueqat.utils import Vqe, QaoaAnsatz, Z


def maxcut_qaoa(n_step: int, edges: List[Tuple[int, int]],
                sampler: Optional[Callable] = None) -> Vqe:
    """Build a Vqe runner for the Max-Cut problem on the given graph.

    :param n_step: Number of QAOA layers (p).
    :param edges: List of (i, j) edges of the graph.
    :param sampler: Optional custom sampler, passed through to Vqe.
    :returns: A Vqe instance, ready to call `.run()` on.
    """
    hamiltonian = 0
    for i, j in edges:
        hamiltonian = hamiltonian + Z[i] * Z[j]

    ansatz = QaoaAnsatz(hamiltonian, n_step)
    return Vqe(ansatz, sampler=sampler)


if __name__ == "__main__":
    print("Max-Cut via QAOA")
    print("=" * 50)

    graph_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (0, 2), (4, 0), (4, 3)]
    runner = maxcut_qaoa(2, graph_edges)

    result = runner.run(max_iter=300)
    best_config = result.most_common()[0][0]

    print(f"Best partition (q0..q4) = {best_config}")
    print("""
         {4}
        / \\
       {0}---{3}
       | x |
       {1}---{2}
""".format(*best_config))
