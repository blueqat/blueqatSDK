blueqat documentation
=====================

**blueqat** is an open-source Python SDK for quantum computing, built natively
on PyTorch. Circuits run on a differentiable statevector / tensor-network
simulator, so gradients of quantum programs (for VQE, QAOA, pulse
optimization, ...) come for free through autograd.

.. code-block:: python

   from blueqat import Circuit

   # A Bell pair, sampled 100 times
   Circuit(2).h[0].cx[0, 1].m[:].run(shots=100)
   # => Counter({'00': 52, '11': 48})

Highlights
----------

- **Two execution modes** behind one API: dense ``statevector`` and
  ``tensornet`` (tensor-network contraction, the default) for large circuits.
- **Differentiable end to end**: gate parameters can be
  ``torch.Tensor`` values with ``requires_grad=True``.
- **Exchange-only spin qubits** (:mod:`blueqat.eo`): encode logical qubits in
  3 spins and compile circuits to Heisenberg exchange pulses, including
  differentiable pulse synthesis and hardware-facing pulse schedules.
- **Interop**: OpenQASM 2.0 input/output, versioned JSON circuit
  serialization, circuit drawing.
- **Cloud groundwork** (:mod:`blueqat.cloud`): API-key management and a
  ``backend='cloud'`` submission path.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   guide/circuits
   guide/backends
   guide/autograd
   guide/exchange_only
   guide/cloud

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
