Differentiable circuits, VQE and QAOA
=====================================

Gradients through the simulator
-------------------------------

Any gate parameter may be a :class:`torch.Tensor` with
``requires_grad=True``. The whole pipeline -- gate matrices, state
propagation (in both execution modes), probabilities, expectation values --
is built from differentiable torch operations:

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import Z

   theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
   energy = Circuit(1).rx(theta)[0].expect(1.0 * Z[0])
   energy.backward()
   theta.grad        # -sin(0.4), the exact analytic gradient

This means variational algorithms need no parameter-shift rule: plain
``torch.optim`` optimizers work directly.

Pauli operators and Hamiltonians
--------------------------------

:mod:`blueqat.utils` provides the Pauli algebra:

.. code-block:: python

   from blueqat.utils import X, Y, Z, I, from_qubo, qubo_bit

   h = 0.5 * Z[0] * Z[1] + 1.2 * X[0] - 3.0
   h = h.simplify()
   h.to_matrix(2)                   # dense or sparse torch matrix
   term = (X[0] * Y[1]).to_term()
   evo = term.get_time_evolution()  # appends exp(-i t P) to a circuit

``from_qubo`` converts a QUBO cost matrix into an Ising Hamiltonian.

VQE
---

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import AnsatzBase, Vqe, Z, X

   class MyAnsatz(AnsatzBase):
       def get_circuit(self, params):
           return Circuit(2).rx(params[0])[0].ry(params[1])[1].cx[0, 1]

   hamiltonian = (1.0 * Z[0] * Z[1] + 0.5 * X[0]).simplify()
   ansatz = MyAnsatz(hamiltonian, n_params=2)
   result = Vqe(ansatz).run()
   result.most_common(4)

``Vqe`` accepts any ``torch.optim`` optimizer class, an optional sampler
(e.g. ``get_measurement_sampler(n)`` for shot-based estimation or
``non_sampling_sampler`` for exact, gradient-preserving expectation), and
``initial_params``.

QAOA
----

:class:`~blueqat.utils.QaoaAnsatz` builds the standard QAOA ansatz from a
Hamiltonian whose terms must mutually commute (checked automatically):

.. code-block:: python

   from blueqat.utils import QaoaAnsatz, Vqe, from_qubo

   qubo = [[1, 1], [1, 0]]
   h = from_qubo(qubo)
   ansatz = QaoaAnsatz(h.simplify(), step=2)
   result = Vqe(ansatz).run()
   print(result.most_common(2))

See ``examples/maxcut_qaoa.py`` and ``examples/vqe_ground_state.py`` in the
repository for complete, self-verifying programs.
