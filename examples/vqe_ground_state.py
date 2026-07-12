"""VQE basics: find the ground-state energy of a single-qubit Hamiltonian.

This shows the general `AnsatzBase` + `Vqe` API (not tied to QAOA): subclass
`AnsatzBase`, implement `get_circuit(params)`, and let `Vqe` optimize the
parameters with PyTorch autograd to minimize <psi|H|psi>.

For H = a*Z + b*X, the exact ground-state energy is -sqrt(a**2 + b**2),
which this script checks the VQE result against.
"""
import math

import torch

from blueqat import Circuit
from blueqat.utils import AnsatzBase, Vqe, X, Z


class SingleQubitAnsatz(AnsatzBase):
    """A generic single-qubit ansatz: an RY rotation followed by an RZ
    rotation can reach any point on the Bloch sphere (up to global phase)."""

    def get_circuit(self, params: torch.Tensor) -> Circuit:
        return Circuit(1).ry(params[0])[0].rz(params[1])[0]


if __name__ == "__main__":
    print("VQE ground-state search")
    print("=" * 50)

    a, b = 0.5, 0.8
    hamiltonian = (a * Z[0] + b * X[0]).simplify()
    print(f"Hamiltonian: {hamiltonian}")

    ansatz = SingleQubitAnsatz(hamiltonian, n_params=2)
    vqe = Vqe(ansatz)
    result = vqe.run(max_iter=300)

    found_energy = result.circuit.run(hamiltonian=hamiltonian).item()
    exact_energy = -math.sqrt(a ** 2 + b ** 2)

    print(f"Optimized parameters: {result.params}")
    print(f"VQE energy:   {found_energy:.6f}")
    print(f"Exact energy: {exact_energy:.6f}")
