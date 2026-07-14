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

from ..gate import *
from .backendbase import Backend


class QasmOutputBackend(Backend):
    """Backend for OpenQASM output."""
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        def _parse_run_args(output_prologue=True, **_kwargs):
            return {'output_prologue': output_prologue}

        args = _parse_run_args(*args, **kwargs)
        if args['output_prologue']:
            qasmlist = [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                f"qreg q[{n_qubits}];",
                f"creg c[{n_qubits}];",
            ]
        else:
            qasmlist = []
        return gates, (qasmlist, n_qubits)

    def _postprocess_run(self, ctx):
        return "\n".join(ctx[0])

    def _one_qubit_gate_noargs(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername} q[{idx}];")
        return ctx

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs
    gate_sx = _one_qubit_gate_noargs
    gate_sxdg = _one_qubit_gate_noargs

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername} q[{control}],q[{target}];")
        return ctx

    gate_cz = _two_qubit_gate_noargs
    gate_cx = _two_qubit_gate_noargs
    gate_cy = _two_qubit_gate_noargs
    gate_ch = _two_qubit_gate_noargs
    gate_swap = _two_qubit_gate_noargs

    def _one_qubit_gate_args_theta(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername}({gate.theta}) q[{idx}];")
        return ctx

    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def gate_i(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"id q[{idx}];")
        return ctx

    def gate_u(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            if abs(gate.gamma) > 1e-7:
                ctx[0].append(
                    f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam}) q[{idx}]; // global phase e^i{gate.gamma} is ignored."
                )
            else:
                ctx[0].append(
                    f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam}) q[{idx}];"
                )
        return ctx

    def gate_phase(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(
                f"p({gate.theta}) q[{idx}];")
        return ctx

    def gate_cu(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(
                f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam},{gate.gamma}) q[{c}],q[{t}];"
            )
        return ctx

    def gate_cphase(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"cp({gate.theta}) q[{c}],q[{t}];")
        return ctx

    def _two_qubit_gate_args_theta(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername}({gate.theta}) q[{c}],q[{t}];")
        return ctx

    gate_crx = _two_qubit_gate_args_theta
    gate_cry = _two_qubit_gate_args_theta
    gate_crz = _two_qubit_gate_args_theta
    gate_rxx = _two_qubit_gate_args_theta
    gate_ryy = _two_qubit_gate_args_theta
    gate_rzz = _two_qubit_gate_args_theta

    def gate_zz(self, gate, ctx):
        # ZZ = diag(1, i, i, 1) = e^{i pi/4} * rzz(pi/2); QASM 2.0 can't express
        # the global phase, so it's noted and dropped (same policy as gate_u).
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"rzz(pi/2) q[{c}],q[{t}]; // global phase e^(i*pi/4) is ignored.")
        return ctx

    def gate_zzdg(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"rzz(-pi/2) q[{c}],q[{t}]; // global phase e^(-i*pi/4) is ignored.")
        return ctx

    def _three_qubit_gate_noargs(self, gate, ctx):
        c0, c1, t = gate.targets
        ctx[0].append(f"{gate.lowername} q[{c0}],q[{c1}],q[{t}];")
        return ctx

    gate_ccx = _three_qubit_gate_noargs
    gate_cswap = _three_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"measure q[{idx}] -> c[{idx}];")
        return ctx

    gate_reset = _one_qubit_gate_noargs
