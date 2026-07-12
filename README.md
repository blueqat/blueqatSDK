# blueqat
A Quantum Computing SDK

Blueqat's simulator is built on PyTorch, with two selectable execution modes:
a dense **statevector** simulator and a memory-scalable **tensornet**
(tensor-network contraction) simulator, which is the default. Both are
differentiable, so circuits with `torch.Tensor` parameters keep their
gradients through `Circuit.run()`.

### Tutorial
https://github.com/Blueqat/Blueqat-tutorials

### Examples
Runnable scripts in [`examples/`](examples/):
- `bell_state.py` -- circuit basics: statevector, single amplitude, shot sampling
- `vqe_ground_state.py` -- VQE with a custom `AnsatzBase` (not tied to QAOA)
- `maxcut_qaoa.py` -- QAOA for the graph Max-Cut problem
- `numpartition_qaoa.py` -- QAOA for number partitioning

### Install
```
git clone https://github.com/blueqat/blueqatSDK
cd blueqatSDK
pip install -e .
```

### Circuit
```python
from blueqat import Circuit
import math

#number of qubit is not specified
c = Circuit()

#if you want to specified the number of qubit
c = Circuit(50) #50qubits
```

### Method Chain
```python
# write as chain
Circuit().h[0].x[0].z[0]

# write in separately
c = Circuit().h[0]
c.x[0].z[0]
```

### Slice
```python
Circuit().z[1:3] # Zgate on 1,2
Circuit().x[:3] # Xgate on (0, 1, 2)
Circuit().h[:] # Hgate on all qubits
Circuit().x[1, 2] # 1qubit gate with comma
```

### Rotation Gate
```python
Circuit().rz(math.pi / 4)[0]
```

### Run
```python
from blueqat import Circuit
Circuit(20).h[:].run() # returns a torch.Tensor statevector

# Select the execution mode explicitly (tensornet is the default)
Circuit(20).h[:].run(mode="statevector")
Circuit(20).h[:].run(mode="tensornet")
```

### Run(shots=n)
```python
Circuit(100).x[:].run(shots=1)
# => Counter({'1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111': 1})
```

### Large-scale circuits
The dense statevector has `2**n_qubits` entries, so for large `n_qubits` in
`tensornet` mode (the default), `run()` requires either `shots=` or
`returns="amplitude"` instead of materializing the full vector:
```python
Circuit(50).h[:].run(shots=3)
Circuit(50).h[:].run(returns="amplitude", amplitude="0" * 50)
```

### Single Amplitude
```python
Circuit(4).h[:].run(amplitude="0101")
```

### Reset and mid-circuit measurement
```python
# reset[i] forces qubit i back to |0>. Any circuit containing reset is run
# shot-by-shot with a real probabilistic collapse at each measure/reset.
Circuit(2).h[0].cx[0, 1].reset[0].m[:].run(shots=100)

# Measurement keys let you tag a measurement and read it back per-shot.
Circuit().x[0].m(key="a")[0].run(shots=10, returns="samples")
# => [{'a': [1]}, {'a': [1]}, ...]
```

### Ancilla qubits
```python
c = Circuit(4).h[0].h[1].h[2].h[3]
with c.ancilla() as a:       # allocate a fresh qubit past the current width
    c.cx[0, a[0]]
    c.cx[0, a[0]]
with c.ancilla(pos=6, stop=8, reset=True) as a:  # or pin an explicit range
    c.cx[3, a[0]]
# a[i] is reset back to |0> on exiting the `with` block when reset=True (the default)
```

### Expectation value of hamiltonian
```python
from blueqat.utils import Z
hamiltonian = 1*Z[0]+1*Z[1]
Circuit(4).x[:].run(hamiltonian=hamiltonian)
# => -2.0
```

### Blueqat to/from QASM
```python
Circuit().h[0].to_qasm()

#OPENQASM 2.0;
#include "qelib1.inc";
#qreg q[1];
#creg c[1];
#h q[0];

from blueqat.circuit_funcs import from_qasm
from_qasm(Circuit().h[0].to_qasm())  # parses back into an equivalent Circuit
```

### Hamiltonian
```python
from blueqat.utils import X, Y, Z, I

h1 = 1.23 * Z[0] + 4.56 * X[1] * Z[2]
h2 = 2.46 * Y[0] + 5.55 * Z[1] * X[2] * X[1]
hamiltonian = h1 * h1 + h2 * h2
print(hamiltonian)
```

### Simplify the Hamiltonian
```python
hamiltonian = hamiltonian.simplify()
print(hamiltonian)
```

### QUBO Hamiltonian
```python
from blueqat.utils import qubo_bit as q

hamiltonian = -3*q(0)-3*q(1)-3*q(2)-3*q(3)-3*q(4)+2*q(0)*q(1)+2*q(0)*q(2)+2*q(0)*q(3)+2*q(0)*q(4)
print(hamiltonian)
```

### Time Evolution
```python
import numpy as np
from blueqat import Circuit
from blueqat.utils import Z, X

hamiltonian = [1.0*Z[0], 1.0*X[0]]
a = [term.get_time_evolution() for term in hamiltonian]

time_evolution = Circuit().h[0]
for evo in a:
    evo(time_evolution, np.random.rand())

print(time_evolution)
```

### VQE
```python
import torch
from blueqat import Circuit
from blueqat.utils import Z, AnsatzBase, Vqe

class MyAnsatz(AnsatzBase):
    def get_circuit(self, params: torch.Tensor) -> Circuit:
        return Circuit(1).rx(params[0])[0]

hamiltonian = 1.0 * Z[0]
vqe = Vqe(MyAnsatz(hamiltonian, n_params=1))
result = vqe.run(initial_params=torch.tensor([0.1]))  # initial_params is optional
print(result.params, result.circuit.run())
print(vqe.sampler_call_count)  # 0 unless a sampler was supplied to Vqe(...)
```

### QAOA
```python
from blueqat.utils import qubo_bit as q, QaoaAnsatz, Vqe

hamiltonian = q(0)-q(1)
step = 1

vqe = Vqe(QaoaAnsatz(hamiltonian, step))
result = vqe.run()
result.circuit.run(shots=100)

# => Counter({'10': 100})
```

### Drawing
```python
Circuit().h[0].cx[0, 1].m[:].run(backend="draw")     # circuit diagram
Circuit().h[0].cx[0, 1].run(backend="draw_tn")        # tensor-network graph
```

### Document
https://blueqat.readthedocs.io/en/latest/


### Disclaimer
Copyright 2026 The blueqat Developers.
