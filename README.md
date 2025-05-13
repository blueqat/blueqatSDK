![Logo](https://raw.githubusercontent.com/Blueqat/Blueqat/master/blueqat_logo_blue.png)

# blueqat
A Quantum Computing SDK

### Version
[![Version](https://badge.fury.io/py/blueqat.svg)](https://badge.fury.io/py/blueqat)

### Tutorial
https://github.com/Blueqat/Blueqat-tutorials

### Notice
The back end has been changed to tensor network. The previous backend environment can still be used with .run(backend="numpy").

### Install
```
git clone https://github.com/Blueqat/Blueqat
cd Blueqat
pip3 install -e .
```

or

```
pip3 install blueqat
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
Circuit(50).h[:].run()
```

### Run(shots=n)
```python
Circuit(100).x[:].run(shots=100)
# => Counter({'1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111': 100})
```

### Single Amplitude
```python
Circuit(4).h[:].run(amplitude="0101")
```

### Expectation value of hamiltonian
```python
from blueqat.pauli import Z
hamiltonian = 1*Z[0]+1*Z[1]
Circuit(4).x[:].run(hamiltonian=hamiltonian)
# => -2.0
```

### Blueqat to QASM
```python
Circuit().h[0].to_qasm()
    
#OPENQASM 2.0;
#include "qelib1.inc";
#qreg q[1];
#creg c[1];
#h q[0];
```

### Hamiltonian
```python
from blueqat.pauli import *

hamiltonian1 = (1.23 * Z[0] + 4.56 * X[1] * Z[2]) ** 2
hamiltonian2 = (2.46 * Y[0] + 5.55 * Z[1] * X[2] * X[1]) ** 2
hamiltonian = hamiltonian1 + hamiltonian2
print(hamiltonian)
    
# => 7.5645*I + 5.6088*Z[0]*X[1]*Z[2] + 5.6088*X[1]*Z[2]*Z[0] + 20.793599999999998*X[1]*Z[2]*X[1]*Z[2] + 13.652999999999999*Y[0]*Z[1]*X[2]*X[1] + 13.652999999999999*Z[1]*X[2]*X[1]*Y[0] + 30.8025*Z[1]*X[2]*X[1]*Z[1]*X[2]*X[1]
```

### Simplify the Hamiltonian
```python
hamiltonian = hamiltonian.simplify()
print(hamiltonian)

#=>-2.4444000000000017*I + 27.305999999999997j*Y[0]*Y[1]*X[2] + 11.2176*Z[0]*X[1]*Z[2]
```

### QUBO Hamiltonian
```python
from blueqat.pauli import qubo_bit as q

hamiltonian = -3*q(0)-3*q(1)-3*q(2)-3*q(3)-3*q(4)+2*q(0)*q(1)+2*q(0)*q(2)+2*q(0)*q(3)+2*q(0)*q(4)
print(hamiltonian)
    
# => -5.5*I + 1.0*Z[1] + 1.0*Z[2] + 1.0*Z[3] + 1.0*Z[4] + 0.5*Z[0]*Z[1] + 0.5*Z[0]*Z[2] + 0.5*Z[0]*Z[3] - 0.5*Z[0] + 0.5*Z[0]*Z[4]
```

### Time Evolution
```python
hamiltonian = [1.0*Z(0), 1.0*X[0]]
a = [term.get_time_evolution() for term in hamiltonian]

time_evolution = Circuit().h[0]
for evo in a:
    evo(time_evolution, np.random.rand())
    
print(time_evolution)

# => Circuit(1).h[0].rz(-1.4543063361067243)[0].h[0].rz(-1.8400416676737137)[0].h[0]
```

### QAOA
```python
from blueqat import Circuit
from blueqat.utils import qaoa
from blueqat.pauli import qubo_bit as q
from blueqat.pauli import X,Y,Z,I

hamiltonian = q(0)-q(1)
step = 1

result = qaoa(hamiltonian, step)
result.circuit.run(shots=100)
    
# => Counter({'01': 99, '11': 1})
```


### Circuit Drawing Backend
```python
from blueqat import vqe
from blueqat.pauli import *
from blueqat.pauli import qubo_bit as q

#hamiltonian = q(0)-3*q(1)+2*q(0)*q(1)+3*q(2)*q(3)+q(4)*q(7)
hamiltonian = Z[0]-3*Z[1]+2*Z[0]*Z[1]+3*Z[2]*Z[3]+Z[4]
step = 8

result = vqe.Vqe(vqe.QaoaAnsatz(hamiltonian, step)).run()
result.circuit.run(backend='draw')
```

![draw](https://raw.githubusercontent.com/Blueqat/Blueqat/master/draw.png)

### Cloud System Connection (API Key is required)
```python
from bqcloud import register_api
api = register_api("Your API Key")

from bqcloud import load_api
api = load_api()

from blueqat import Circuit
from bqcloud import Device

task = api.execute(Circuit().h[0].cx[0, 1], Device.IonQDevice, 10)
#task = api.execute(Circuit().h[0].cx[0, 1], Device.AspenM1, 10)

# Wait 10 sec. If complete, result is returned, otherwise, None is returned.
result = task.wait(timeout=10)

if result:
    print(result.shots())
else:
    print("timeout")
```

### Document
https://blueqat.readthedocs.io/en/latest/

### Contributors
[Contributors](https://github.com/Blueqat/Blueqat/graphs/contributors)

### Disclaimer
Copyright 2022 The Blueqat Developers.
