# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from blueqat import Circuit

QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.23) q[2];
x q[2];
y q[2];
cz q[2],q[1];
z q[1];
ry(4.56) q[0];
u(1.0,2.0,3.0) q[0];
cu(2.0,3.0,1.0,0.5) q[2],q[0];
reset q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];"""

def test_qasm1():
    c = Circuit()
    c.h[0, 1].cx[0, 1].rz(1.23)[2].x[2].y[2].cz[2, 1].z[1].ry(4.56)[0]
    c.u(1.0, 2.0, 3.0)[0]
    c.cu(2.0, 3.0, 1.0, 0.5)[2, 0]
    c.reset[1]
    qasm = c.m[:].to_qasm()
    assert QASM == qasm

def qasm_prologue(n_qubits):
    return "\n".join([
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[" + str(n_qubits) + "];",
        "creg c[" + str(n_qubits) + "];"
    ])

def test_qasm_nocache():
    correct_qasm = qasm_prologue(1) + "\nx q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    c.run()
    c.to_qasm()
    qasm = c.to_qasm()
    assert qasm == correct_qasm

def test_qasm_noprologue():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(output_prologue=False)
    assert qasm == correct_qasm

def test_qasm_noprologue2():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(False)
    assert qasm == correct_qasm
