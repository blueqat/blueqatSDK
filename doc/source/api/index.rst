API reference
=============

Core
----

.. automodule:: blueqat.circuit
   :members: Circuit, BlueqatGlobalSetting
   :undoc-members:

.. automodule:: blueqat.gate
   :members:
   :undoc-members:
   :exclude-members: slicing, slicing_singlevalue, qubit_pairs, get_maximum_index

Pauli operators, VQE and QAOA
-----------------------------

.. automodule:: blueqat.utils
   :members:
   :undoc-members:

Backends
--------

.. automodule:: blueqat.backends.backendbase
   :members:

.. automodule:: blueqat.backends.torch_backend
   :members: TorchBackend

Exchange-only spin qubits
-------------------------

.. automodule:: blueqat.eo.encoding
   :members:

.. automodule:: blueqat.eo.sequences
   :members:

.. automodule:: blueqat.eo.optimizer
   :members:

.. automodule:: blueqat.eo.schedule
   :members:

.. automodule:: blueqat.eo.transpiler
   :members:

Cloud
-----

.. automodule:: blueqat.cloud
   :members:

Circuit utilities
-----------------

.. automodule:: blueqat.circuit_funcs.qasm_parser
   :members:

.. automodule:: blueqat.circuit_funcs.json_serializer
   :members: serialize, deserialize

.. automodule:: blueqat.circuit_funcs.circuit_to_unitary
   :members:

.. automodule:: blueqat.circuit_funcs.flatten
   :members:
