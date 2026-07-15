Circuits and gates
==================

Building circuits
-----------------

:class:`~blueqat.circuit.Circuit` stores a list of operations. Gates are
attributes; qubits are selected with ``[...]``; parametric gates take their
parameters as a call before the qubit indexing. Everything chains:

.. code-block:: python

   import math
   from blueqat import Circuit

   Circuit().h[0].cx[0, 1].rz(math.pi / 4)[1].m[:]

Qubit 0 is always the least-significant bit of the statevector index
(``'10'`` means qubit 1 is 1, qubit 0 is 0 -- the same convention as Qiskit's
``Statevector``).

Gate set
--------

Single-qubit gates
   ``i``, ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``,
   ``sxdg``, ``phase(theta)`` (aliases ``p``, ``r``), ``rx(theta)``,
   ``ry(theta)``, ``rz(theta)``, ``u(theta, phi, lam[, gamma])``,
   ``mat1(matrix)`` (arbitrary 2x2 unitary).

Two-qubit gates
   ``cx`` (alias ``cnot``), ``cy``, ``cz``, ``ch``, ``swap``, ``iswap``,
   ``iswapdg``, ``cphase(theta)`` (aliases ``cp``, ``cr``), ``crx``, ``cry``,
   ``crz``, ``cu(theta, phi, lam[, gamma])``, ``rxx(theta)``, ``ryy(theta)``,
   ``rzz(theta)``, ``zz``, ``zzdg``, ``exch(theta)`` (Heisenberg exchange
   pulse, see :doc:`exchange_only`).

Three-qubit gates
   ``ccx`` (alias ``toffoli``), ``ccz``, ``cswap``.

Other operations
   ``m`` / ``measure`` (optionally ``m(key="name")`` for keyed mid-circuit
   measurement), ``reset``, ``barrier``.

Gates that take no parameters raise ``ValueError`` if parameters are passed
(e.g. ``x(0.5)[0]`` is rejected rather than silently ignored).

Introspection
-------------

.. code-block:: python

   c = Circuit(3).h[:].cx[0, 1].cx[1, 2].m[:]
   c.n_qubits      # 3
   c.depth()       # 4  (parallel gates count once; barriers don't count)
   c.count_ops()   # Counter({'h': 3, 'cx': 2, 'measure': 3})

Measurement probabilities (differentiable, optionally marginalized onto
selected qubits) and Hamiltonian expectation values:

.. code-block:: python

   from blueqat.utils import Z

   Circuit(2).h[0].cx[0, 1].probs()          # tensor([0.5, 0., 0., 0.5])
   Circuit(2).h[0].cx[0, 1].probs([1])       # marginal of qubit 1
   Circuit(1).rx(0.4)[0].expect(1.0 * Z[0])  # <Z> = cos(0.4)

Inverse circuits
----------------

:meth:`~blueqat.circuit.Circuit.dagger` returns the Hermitian conjugate
(gates reversed and conjugated). Measurement and reset have no inverse;
``dagger(ignore_measurement=True)`` drops them instead of raising:

.. code-block:: python

   c = Circuit(3)  # ... build ...
   identity = c + c.dagger()   # uncomputes back to |0...0>

OpenQASM 2.0
------------

.. code-block:: python

   qasm = Circuit(2).h[0].cx[0, 1].to_qasm()

   from blueqat.circuit_funcs import from_qasm
   c = from_qasm(qasm)

JSON serialization
------------------

Circuits round-trip through a versioned, JSON-compatible schema (this is also
the cloud submission wire format):

.. code-block:: python

   from blueqat.circuit_funcs.json_serializer import serialize, deserialize

   data = serialize(Circuit(2).h[0].cx[0, 1])
   c = deserialize(data)

Drawing
-------

``run(backend='draw')`` renders the circuit with matplotlib. Every registered
gate is drawable; unknown (user-registered) gates are omitted with a
``UserWarning``.

Ancilla qubits
--------------

.. code-block:: python

   c = Circuit(4).h[:]
   with c.ancilla() as a:        # allocates a fresh qubit
       c.cx[0, a[0]]
       c.cx[0, a[0]]
   # the ancilla is reset to |0> on exit (reset=True by default)

Macros and custom gates
-----------------------

Register a function as a circuit method, or a gate class into the gate set:

.. code-block:: python

   from blueqat import BlueqatGlobalSetting
   from blueqat.decorators import circuitmacro

   @circuitmacro
   def bell(c, a, b):
       return c.h[a].cx[a, b]

   Circuit(2).bell(0, 1)

   BlueqatGlobalSetting.register_gate('mygate', MyGateClass)
