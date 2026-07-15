Backends and execution
======================

Simulation modes
----------------

One simulator, two execution modes:

- ``tensornet`` (default): tensor-network contraction via ``opt_einsum``.
  Never materializes the full state unless asked to, so wide-but-shallow
  circuits scale far beyond dense simulation.
- ``statevector``: dense statevector propagation.

.. code-block:: python

   Circuit(20).h[:].run()                      # tensornet (default)
   Circuit(20).h[:].run(backend='statevector') # dense
   Circuit(20).h[:].run(mode='statevector')    # equivalent

Both modes agree numerically and both preserve autograd graphs.

Return values
-------------

.. code-block:: python

   c = Circuit(2).h[0].cx[0, 1]

   c.run()                                   # statevector (torch.Tensor)
   c.statevector()                           # same, explicit
   c.m[:].run(shots=100)                     # Counter of bitstrings
   c.shots(100)                              # same, explicit
   c.run(amplitude='11')                     # a single amplitude
   c.m[:].oneshot()                          # (collapsed state, one outcome)
   c.expect(hamiltonian)                     # <psi|H|psi>
   c.probs([1])                              # marginal probabilities

Large circuits
--------------

The dense state has ``2**n`` entries. In ``tensornet`` mode, circuits with
more than 28 qubits require ``shots=`` or ``returns='amplitude'`` instead of
the full vector:

.. code-block:: python

   Circuit(50).h[:].run(shots=3)
   Circuit(50).h[:].run(returns='amplitude', amplitude='0' * 50)

Sampling uses inverse-CDF search, so there is no category-count limit.

Mid-circuit measurement and reset
---------------------------------

``reset`` and keyed measurement make outcomes depend on when the collapse
happens, so such circuits automatically run shot-by-shot as quantum
trajectories, collapsing at each ``measure`` / ``reset``:

.. code-block:: python

   Circuit(2).h[0].cx[0, 1].reset[0].m[:].run(shots=100)

   Circuit().x[0].m(key='a')[0].run(shots=3, returns='samples')
   # [{'a': [1]}, {'a': [1]}, {'a': [1]}]

Custom initial states
---------------------

.. code-block:: python

   import torch
   psi0 = torch.tensor([0, 1, 0, 0], dtype=torch.complex128)
   Circuit(2).h[0].run(initial=psi0)

Other built-in backends
-----------------------

- ``'draw'`` -- matplotlib circuit diagram.
- ``'draw_tn'`` -- the tensor-network graph of the circuit.
- ``'eo'`` -- exchange-only transpiler (see :doc:`exchange_only`).
- ``'cloud'`` -- cloud submission (see :doc:`cloud`).
- ``'1q_compaction'`` / ``'2q_decomposition'`` -- transpilers merging
  single-qubit gates / rewriting two-qubit gates into a chosen basis.

Registering your own backend
----------------------------

.. code-block:: python

   from blueqat import register_backend, Backend

   class MyBackend(Backend):
       def run(self, gates, n_qubits, *args, **kwargs):
           ...

   register_backend('mybackend', MyBackend)
   Circuit(2).h[0].run(backend='mybackend')
   Circuit(2).h[0].run_with_mybackend()      # equivalent
