Getting started
===============

Installation
------------

blueqat requires Python 3.11+ and installs PyTorch as its simulation core:

.. code-block:: console

   pip install git+https://github.com/blueqat/blueqatSDK

For development:

.. code-block:: console

   git clone https://github.com/blueqat/blueqatSDK
   cd blueqatSDK
   pip install -e .[dev]
   pytest tests/ -q

First circuit
-------------

Circuits are built by method chaining. A gate is selected as an attribute and
applied to qubits with ``[...]`` indexing:

.. code-block:: python

   from blueqat import Circuit

   c = Circuit()            # width grows automatically
   c.h[0]                   # Hadamard on qubit 0
   c.cx[0, 1]               # CNOT: control 0, target 1

   # or equivalently, as one chain:
   c = Circuit().h[0].cx[0, 1]

Running it returns the statevector as a :class:`torch.Tensor` (qubit 0 is the
least-significant bit of the state index):

.. code-block:: python

   c.run()
   # tensor([0.7071+0.j, 0.0000+0.j, 0.0000+0.j, 0.7071+0.j])

Sampling measurement outcomes instead:

.. code-block:: python

   c.m[:].run(shots=1000)
   # Counter({'00': 493, '11': 507})

Slices apply a gate to many qubits at once:

.. code-block:: python

   Circuit(4).h[:]          # H on every qubit
   Circuit(4).x[1:3]        # X on qubits 1, 2
   Circuit(4).z[0, 3]       # Z on qubits 0 and 3

Where to go next
----------------

- :doc:`guide/circuits` -- the full gate set, circuit introspection, QASM.
- :doc:`guide/backends` -- statevector vs tensornet, shots, large circuits.
- :doc:`guide/autograd` -- differentiable circuits, VQE and QAOA.
- :doc:`guide/exchange_only` -- exchange-only spin qubits and pulse
  compilation.
