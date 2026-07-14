# Copyright 2019-2026 The Blueqat Developers
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
"""
This module manages the set of operations, and provides a factory method of operations.
Maintained for PyTorch Tensor Network integration in 2026.
"""

from typing import Dict, Optional, Type

from . import gate
from .typing import Targets

GATE_SET: Dict[str, Type[gate.Operation]] = {
    # 1 qubit gates (alphabetical)
    "h": gate.HGate,
    "i": gate.IGate,
    "mat1": gate.Mat1Gate,
    "p": gate.PhaseGate,
    "phase": gate.PhaseGate,
    "r": gate.PhaseGate,
    "rx": gate.RXGate,
    "ry": gate.RYGate,
    "rz": gate.RZGate,
    "s": gate.SGate,
    "sdg": gate.SDagGate,
    "sx": gate.SXGate,
    "sxdg": gate.SXDagGate,
    "t": gate.TGate,
    "tdg": gate.TDagGate,
    "u": gate.UGate,
    "x": gate.XGate,
    "y": gate.YGate,
    "z": gate.ZGate,
    # Controlled gates (alphabetical)
    "ccx": gate.ToffoliGate,
    "ccz": gate.CCZGate,
    "cnot": gate.CXGate,
    "ch": gate.CHGate,
    "cp": gate.CPhaseGate,
    "cphase": gate.CPhaseGate,
    "cr": gate.CPhaseGate,
    "crx": gate.CRXGate,
    "cry": gate.CRYGate,
    "crz": gate.CRZGate,
    "cswap": gate.CSwapGate,
    "cu": gate.CUGate,
    "cx": gate.CXGate,
    "cy": gate.CYGate,
    "cz": gate.CZGate,
    "toffoli": gate.ToffoliGate,
    # Other multi qubit gates (alphabetical)
    "iswap": gate.ISwapGate,
    "iswapdg": gate.ISwapDagGate,
    "rxx": gate.RXXGate,
    "ryy": gate.RYYGate,
    "rzz": gate.RZZGate,
    "swap": gate.SwapGate,
    "zz": gate.ZZGate,
    "zzdg": gate.ZZDagGate,
    # Measure, reset and barrier (alphabetical)
    "barrier": gate.Barrier,
    "m": gate.Measurement,
    "measure": gate.Measurement,
    "reset": gate.Reset,
}


def get_op_type(name: str) -> Optional[Type[gate.Operation]]:
    """Get a class of operation from operation name."""
    return GATE_SET.get(name)


def create(name: str,
           targets: Targets,
           params: tuple,
           options: Optional[dict] = None) -> gate.Operation:
    """Create an operation from name, targets and params."""
    op_type = get_op_type(name)
    if op_type is None:
        raise ValueError(f"Unknown operation `{name}`.")
    return op_type.create(targets, params, options)


def register_operation(name: str, op_type: Type[gate.Operation]) -> None:
    """Register an operation. If operation already exists, overwrite it."""
    GATE_SET[name] = op_type


def unregister_operation(name: str) -> None:
    """Unregister an operation. If operation does not exist, do nothing."""
    try:
        del GATE_SET[name]
    except KeyError:
        pass