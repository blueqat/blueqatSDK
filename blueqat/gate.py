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
`gate` module implements quantum gate operations.
Modernized for PyTorch Tensor Network integration in 2026.
"""

import cmath
import math
from typing import Any, cast, Callable, Iterable, Iterator, List, NoReturn, Optional, Tuple, Type, TypeVar, Union

import torch

from .typing import Targets

_Op = TypeVar('_Op', bound='Operation')


class Operation:
    """Abstract quantum circuit operation class."""

    lowername: str = ''
    """Lower name of the operation."""

    @property
    def uppername(self) -> str:
        """Upper name of the operation."""
        return self.lowername.upper()

    def __init__(self, targets: Targets, params=()) -> None:
        if self.lowername == '':
            raise ValueError(
                f"{self.__class__.__name__}.lowername is not defined.")
        self.params = params
        self.targets = targets

    def target_iter(self, n_qubits: int) -> Iterator[int]:
        """The generator which yields the target qubits."""
        return slicing(self.targets, n_qubits)

    @classmethod
    def create(cls: Type[_Op], targets: Targets, params: tuple,
               options: Optional[dict]) -> _Op:
        """Create an operation."""
        raise NotImplementedError(f"{cls.__name__}.create() is not defined.")

    def _str_args(self) -> str:
        """Returns printable string of args."""
        if not self.params:
            return ''
        return '(' + ', '.join(str(param) for param in self.params) + ')'

    def _str_targets(self) -> str:
        """Returns printable string of targets."""

        def _slice_to_str(obj):
            if isinstance(obj, slice):
                start = '' if obj.start is None else str(obj.start.__index__())
                stop = '' if obj.stop is None else str(obj.stop.__index__())
                if obj.step is None:
                    return f'{start}:{stop}'
                step = str(obj.step.__index__())
                return f'{start}:{stop}:{step}'
            return str(obj.__index__())

        if isinstance(self.targets, tuple):
            return f"[{', '.join(_slice_to_str(target) for target in self.targets)}]"
        return f"[{_slice_to_str(self.targets)}]"

    def __str__(self) -> str:
        str_args = self._str_args()
        str_targets = self._str_targets()
        return f'{self.lowername}{str_args}{str_targets}'


class IFallbackOperation(Operation):
    """The interface of `fallback`"""

    def fallback(self, n_qubits: int) -> List['Operation']:
        """Get alternative operations"""
        raise NotImplementedError(
            f"fallback of {self.lowername} is not defined.")


class Gate(Operation):
    """Abstract quantum gate class."""

    @property
    def n_qargs(self) -> int:
        """Number of qubit arguments of this gate."""
        raise NotImplementedError()

    def dagger(self) -> 'Gate':
        """Returns the Hermitian conjugate of `self`."""
        raise NotImplementedError(
            "Hermitian conjugate of this gate is not provided.")

    def matrix(self) -> torch.Tensor:
        """Returns the matrix of implementations as a PyTorch Tensor."""
        raise NotImplementedError()


class OneQubitGate(Gate):
    """Abstract quantum gate class for 1 qubit gate."""

    @property
    def n_qargs(self) -> int:
        return 1

    def _make_fallback_for_target_iter(
            self, n_qubits: int,
            fallback: Callable[[int], List['Gate']]) -> List['Gate']:
        gates = []
        for t in self.target_iter(n_qubits):
            gates += fallback(t)
        return gates


class TwoQubitGate(Gate):
    """Abstract quantum gate class for 2 qubits gate."""

    @property
    def n_qargs(self):
        return 2

    def control_target_iter(self, n_qubits: int) -> Iterator[Tuple[int, int]]:
        """The generator which yields the tuples of (control, target) qubits."""
        return qubit_pairs(self.targets, n_qubits)

    def _make_fallback_for_control_target_iter(
            self, n_qubits: int,
            fallback: Callable[[int, int], List['Gate']]) -> List['Gate']:
        gates = []
        for c, t in self.control_target_iter(n_qubits):
            gates += fallback(c, t)
        return gates


class HGate(OneQubitGate):
    """Hadamard gate"""
    lowername = "h"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'HGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        val = 1.0 / math.sqrt(2)
        return torch.tensor([[val, val], [val, -val]], dtype=torch.complex128)


class IGate(OneQubitGate, IFallbackOperation):
    """Identity gate"""
    lowername = "i"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'IGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def fallback(self, _):
        return []

    def dagger(self):
        return self

    def matrix(self):
        return torch.eye(2, dtype=torch.complex128)


class Mat1Gate(OneQubitGate):
    """Arbitrary 2x2 matrix gate"""
    lowername = "mat1"

    def __init__(self, targets, mat: torch.Tensor):
        super().__init__(targets, (mat, ))
        self.mat = torch.as_tensor(mat, dtype=torch.complex128)

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'Mat1Gate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        # mish() を .mT に修正し、テンソルの随伴行列を正しく取得
        return Mat1Gate(self.targets, self.mat.mT.conj())

    def matrix(self):
        return self.mat


class PhaseGate(OneQubitGate):
    """Phase gate"""
    lowername = "phase"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'PhaseGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        return PhaseGate(self.targets, -self.theta)

    def matrix(self):
        theta = torch.as_tensor(self.theta, dtype=torch.complex128)
        # ones_like/zeros_like inherit theta's device, unlike a bare
        # torch.tensor(...) literal (always CPU), which would crash torch.stack
        # with a device mismatch if theta lives on e.g. CUDA/MPS.
        one = torch.ones_like(theta)
        zero = torch.zeros_like(theta)
        elements = torch.stack([one, zero, zero, torch.exp(1j * theta)])
        return elements.reshape(2, 2)


class RXGate(OneQubitGate):
    """Rotate-X gate"""
    lowername = "rx"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RXGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        return RXGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        isin_t = -1j * torch.sin(t)
        elements = torch.stack([cos_t, isin_t, isin_t, cos_t])
        return elements.reshape(2, 2)


class RYGate(OneQubitGate):
    """Rotate-Y gate"""
    lowername = "ry"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RYGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        return RYGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        elements = torch.stack([cos_t, -sin_t, sin_t, cos_t])
        return elements.reshape(2, 2)


class RZGate(OneQubitGate):
    """Rotate-Z gate"""
    lowername = "rz"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RZGate':
        if options:
            raise ValueError(f"{cls.__name__} doesn't take options")
        return cls(targets, params[0])

    def dagger(self):
        return RZGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        a = torch.exp(-1j * t)
        b = torch.exp(1j * t)
        zero = torch.zeros_like(t)
        elements = torch.stack([a, zero, zero, b])
        return elements.reshape(2, 2)


class SGate(OneQubitGate, IFallbackOperation):
    """S gate"""
    lowername = "s"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'SGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return SDagGate(self.targets)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [PhaseGate(t, math.pi / 2)])

    def matrix(self):
        return torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex128)


class SDagGate(OneQubitGate, IFallbackOperation):
    """Dagger of S gate"""
    lowername = "sdg"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'SDagGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return SGate(self.targets)

    def fallback(self, n_qubits):
        return self._make_fallback_for_target_iter(
            n_qubits, lambda t: [PhaseGate(t, -math.pi / 2)])

    def matrix(self):
        return torch.tensor([[1, 0], [0, -1j]], dtype=torch.complex128)


class SXGate(OneQubitGate):
    """sqrt(X) gate"""
    lowername = "sx"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'SXGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return SXDagGate(self.targets)

    def matrix(self):
        return 0.5 * torch.tensor([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=torch.complex128)


class SXDagGate(OneQubitGate):
    """sqrt(X)† gate"""
    lowername = "sxdg"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'SXDagGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return SXGate(self.targets)

    def matrix(self):
        return 0.5 * torch.tensor([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=torch.complex128)


class TGate(OneQubitGate, IFallbackOperation):
    """T gate"""
    lowername = "t"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'TGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return TDagGate(self.targets)

    def fallback(self, _):
        return [PhaseGate(self.targets, math.pi / 4)]

    def matrix(self):
        # cmath.exp on a Python complex keeps full double precision; wrapping the
        # angle in torch.tensor(...) first creates a complex64 intermediate (torch's
        # default complex dtype for a scalar built from a Python complex), losing
        # ~8 significant digits before torch.exp ever runs.
        return torch.tensor([[1, 0], [0, cmath.exp(math.pi * 0.25j)]], dtype=torch.complex128)


class TDagGate(OneQubitGate, IFallbackOperation):
    """Dagger of T gate"""
    lowername = "tdg"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'TDagGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return TGate(self.targets)

    def fallback(self, _):
        return [PhaseGate(self.targets, -math.pi / 4)]

    def matrix(self):
        return torch.tensor([[1, 0], [0, cmath.exp(math.pi * -0.25j)]], dtype=torch.complex128)


class ToffoliGate(Gate, IFallbackOperation):
    """Toffoli (CCX) gate"""
    lowername = "ccx"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ToffoliGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        c1, c2, t = self.targets
        return [HGate(t), CCZGate((c1, c2, t)), HGate(t)]

    def matrix(self):
        # targets = (c1, c2, t) uses the same control=least-significant-bit
        # convention as CXGate.matrix() (index = t*4 + c2*2 + c1), so the flipped
        # pair is where both controls are 1 (indices 3 and 7), not 6 and 7.
        m = torch.eye(8, dtype=torch.complex128)
        m[3, 3], m[3, 7] = 0, 1
        m[7, 3], m[7, 7] = 1, 0
        return m


class UGate(OneQubitGate):
    """Arbitrary 1 qubit unitary gate"""
    lowername = "u"

    def __init__(self, targets, theta, phi, lam, gamma=0.0):
        super().__init__(targets, (theta, phi, lam, gamma))
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'UGate':
        return cls(targets, *params)

    def dagger(self):
        return UGate(self.targets, -self.theta, -self.lam, -self.phi, -self.gamma)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128)
        p = torch.as_tensor(self.phi, dtype=torch.complex128)
        l = torch.as_tensor(self.lam, dtype=torch.complex128)
        g = torch.as_tensor(self.gamma, dtype=torch.complex128)

        gphase = torch.exp(1j * g)
        cos_t = torch.cos(0.5 * t)
        sin_t = torch.sin(0.5 * t)

        elements = torch.stack([
            cos_t,
            -torch.exp(1j * l) * sin_t,
            torch.exp(1j * p) * sin_t,
            torch.exp(1j * (p + l)) * cos_t
        ])
        return elements.reshape(2, 2) * gphase


class XGate(OneQubitGate):
    """Pauli's X gate"""
    lowername = "x"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'XGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)


class YGate(OneQubitGate):
    """Pauli's Y gate"""
    lowername = "y"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'YGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)


class ZGate(OneQubitGate):
    """Pauli's Z gate"""
    lowername = "z"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ZGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


class CCZGate(Gate, IFallbackOperation):
    """2-Controlled Z gate"""
    lowername = "ccz"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CCZGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def fallback(self, n_qubits):
        c1, c2, t = self.targets
        return [
            CXGate((c2, t)), TDagGate(t), CXGate((c1, t)), TGate(t),
            CXGate((c2, t)), TDagGate(t), CXGate((c1, t)), TGate(c2),
            TGate(t), CXGate((c1, c2)), TGate(c1), TDagGate(c2), CXGate((c1, c2)),
        ]

    def dagger(self):
        return self

    def matrix(self):
        m = torch.eye(8, dtype=torch.complex128)
        m[7, 7] = -1
        return m


class CHGate(TwoQubitGate):
    """Controlled-H gate"""
    lowername = "ch"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CHGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        a = 1.0 / math.sqrt(2)
        return torch.tensor([
            [1, 0, 0, 0],
            [0, a, 0, a],
            [0, 0, 1, 0],
            [0, a, 0, -a]
        ], dtype=torch.complex128)


class CPhaseGate(TwoQubitGate):
    """Controlled Phase gate"""
    lowername = "cphase"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CPhaseGate':
        return cls(targets, params[0])

    def dagger(self):
        return CPhaseGate(self.targets, -self.theta)

    def matrix(self):
        theta = torch.as_tensor(self.theta, dtype=torch.complex128)
        one = torch.ones_like(theta)
        return torch.diag(torch.stack([one, one, one, torch.exp(1j * theta)]))


class CRXGate(TwoQubitGate):
    """Controlled RX gate"""
    lowername = "crx"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CRXGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRXGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        isin_t = -1j * torch.sin(t)
        zero = torch.zeros_like(t)
        one = torch.ones_like(t)

        elements = torch.stack([
            one, zero, zero, zero,
            zero, cos_t, zero, isin_t,
            zero, zero, one, zero,
            zero, isin_t, zero, cos_t
        ])
        return elements.reshape(4, 4)


class CRYGate(TwoQubitGate):
    """Controlled RY gate"""
    lowername = "cry"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CRYGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRYGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        zero = torch.zeros_like(t)
        one = torch.ones_like(t)

        elements = torch.stack([
            one, zero, zero, zero,
            zero, cos_t, zero, -sin_t,
            zero, zero, one, zero,
            zero, sin_t, zero, cos_t
        ])
        return elements.reshape(4, 4)


class CRZGate(TwoQubitGate):
    """Controlled RZ gate"""
    lowername = "crz"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CRZGate':
        return cls(targets, params[0])

    def dagger(self):
        return CRZGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        a = torch.exp(-1j * t)
        b = torch.exp(1j * t)
        one = torch.ones_like(t)
        return torch.diag(torch.stack([one, a, one, b]))


class CSwapGate(Gate, IFallbackOperation):
    """Controlled SWAP gate"""
    lowername = "cswap"

    @property
    def n_qargs(self):
        return 3

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CSwapGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def fallback(self, n_qubits):
        c, t1, t2 = self.targets
        return [CXGate((t2, t1)), ToffoliGate((c, t1, t2)), CXGate((t2, t1))]

    def matrix(self):
        # targets = (c, t1, t2), same LSB-first convention as ToffoliGate.matrix()
        # (index = t2*4 + t1*2 + c): the swapped pair is where c=1 and exactly one
        # of t1/t2 is set, i.e. indices 3 (t1=1,t2=0) and 5 (t1=0,t2=1).
        m = torch.eye(8, dtype=torch.complex128)
        m[3, 3], m[3, 5] = 0, 1
        m[5, 3], m[5, 5] = 1, 0
        return m


class CUGate(TwoQubitGate):
    """Controlled-U gate"""
    lowername = "cu"

    def __init__(self, targets, theta, phi, lam, gamma=0.0):
        super().__init__(targets, (theta, phi, lam, gamma))
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CUGate':
        return cls(targets, *params)

    def dagger(self):
        return CUGate(self.targets, -self.theta, -self.lam, -self.phi, -self.gamma)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128)
        p = torch.as_tensor(self.phi, dtype=torch.complex128)
        l = torch.as_tensor(self.lam, dtype=torch.complex128)
        g = torch.as_tensor(self.gamma, dtype=torch.complex128)

        cos_t = torch.cos(0.5 * t)
        sin_t = torch.sin(0.5 * t)
        zero = torch.zeros_like(t)
        one = torch.ones_like(t)

        elements = torch.stack([
            one, zero, zero, zero,
            zero, torch.exp(1j * g) * cos_t, zero, -torch.exp(1j * (g + l)) * sin_t,
            zero, zero, one, zero,
            zero, torch.exp(1j * (g + p)) * sin_t, zero, torch.exp(1j * (g + p + l)) * cos_t
        ])
        return elements.reshape(4, 4)


class CXGate(TwoQubitGate):
    """Controlled-X (CNOT) gate"""
    lowername = "cx"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CXGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=torch.complex128)


class CYGate(TwoQubitGate):
    """Controlled-Y gate"""
    lowername = "cy"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CYGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1, 0],
            [0, 1j, 0, 0]
        ], dtype=torch.complex128)


class CZGate(TwoQubitGate):
    """Controlled-Z gate"""
    lowername = "cz"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'CZGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return self

    def matrix(self):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex128)


class RXXGate(TwoQubitGate):
    """Rotate-XX gate"""
    lowername = "rxx"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RXXGate':
        return cls(targets, *params)

    def dagger(self):
        return RXXGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        isin_t = -1j * torch.sin(t)
        zero = torch.zeros_like(t)

        elements = torch.stack([
            cos_t, zero, zero, isin_t,
            zero, cos_t, isin_t, zero,
            zero, isin_t, cos_t, zero,
            isin_t, zero, zero, cos_t
        ])
        return elements.reshape(4, 4)


class RYYGate(TwoQubitGate):
    """Rotate-YY gate"""
    lowername = "ryy"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RYYGate':
        return cls(targets, *params)

    def dagger(self):
        return RYYGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        cos_t = torch.cos(t)
        isin_t = 1j * torch.sin(t)
        zero = torch.zeros_like(t)

        elements = torch.stack([
            cos_t, zero, zero, isin_t,
            zero, cos_t, -isin_t, zero,
            zero, -isin_t, cos_t, zero,
            isin_t, zero, zero, cos_t
        ])
        return elements.reshape(4, 4)


class RZZGate(TwoQubitGate):
    """Rotate-ZZ gate"""
    lowername = "rzz"

    def __init__(self, targets, theta):
        super().__init__(targets, (theta, ))
        self.theta = theta

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'RZZGate':
        return cls(targets, *params)

    def dagger(self):
        return RZZGate(self.targets, -self.theta)

    def matrix(self):
        t = torch.as_tensor(self.theta, dtype=torch.complex128) * 0.5
        a = torch.exp(1j * t)
        return torch.diag(torch.stack([a.conj(), a, a, a.conj()]))


class SwapGate(TwoQubitGate):
    """Swap gate"""
    lowername = "swap"

    def dagger(self):
        return self

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'SwapGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def matrix(self):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.complex128)


class ZZGate(TwoQubitGate):
    """ZZ gate"""
    lowername = "zz"

    def __init__(self, targets):
        super().__init__(targets, ())

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ZZGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        # matrix() is diag(1, 1j, 1j, 1), which is not Hermitian (its own conjugate
        # transpose is diag(1, -1j, -1j, 1) != itself), so the dagger is a distinct gate.
        return ZZDagGate(self.targets)

    def matrix(self):
        return torch.diag(torch.tensor([1, 1j, 1j, 1], dtype=torch.complex128))


class ZZDagGate(TwoQubitGate):
    """Dagger of ZZ gate"""
    lowername = "zzdg"

    def __init__(self, targets):
        super().__init__(targets, ())

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ZZDagGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return ZZGate(self.targets)

    def matrix(self):
        return torch.diag(torch.tensor([1, -1j, -1j, 1], dtype=torch.complex128))


class ISwapGate(TwoQubitGate, IFallbackOperation):
    """iSWAP gate: swaps two qubits and phases the swapped amplitudes by i."""
    lowername = "iswap"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ISwapGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return ISwapDagGate(self.targets)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [SGate(c), SGate(t), HGate(c),
                          CXGate((c, t)), CXGate((t, c)), HGate(t)])

    def matrix(self):
        # Symmetric in its two qubits, so the control/target bit convention
        # doesn't matter: |01> <-> i|10>.
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.complex128)


class ISwapDagGate(TwoQubitGate, IFallbackOperation):
    """Dagger of iSWAP gate."""
    lowername = "iswapdg"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'ISwapDagGate':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def dagger(self):
        return ISwapGate(self.targets)

    def fallback(self, n_qubits):
        return self._make_fallback_for_control_target_iter(
            n_qubits,
            lambda c, t: [HGate(t), CXGate((t, c)), CXGate((c, t)),
                          HGate(c), SDagGate(t), SDagGate(c)])

    def matrix(self):
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.complex128)


class Barrier(IFallbackOperation):
    """Barrier: a no-op marker separating circuit sections (as in Qiskit and
    OpenQASM). Simulation backends treat it as the identity via its empty
    fallback; the QASM output backend emits a real `barrier` statement."""
    lowername = "barrier"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'Barrier':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take parameters")
        return cls(targets)

    def fallback(self, _):
        return []

    def dagger(self):
        return self


class Measurement(Operation):
    """Measurement operation"""
    lowername = "measure"

    def __init__(self, targets: Targets, options: Optional[dict]):
        super().__init__(targets, ())
        if options is None:
            options = {}
        key = options.get("key")
        self.key = str(key) if key is not None else None
        duplicated = options.get("duplicated")
        self.duplicated = str(duplicated) if duplicated is not None else None

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'Measurement':
        if params:
            raise ValueError(f"{cls.__name__} doesn't take params")
        return cls(targets, options)

    def target_iter(self, n_qubits):
        return slicing(self.targets, n_qubits)


class Reset(Operation):
    """Reset operation"""
    lowername = "reset"

    @classmethod
    def create(cls, targets: Targets, params: tuple, options: Optional[dict] = None) -> 'Reset':
        if params or options:
            raise ValueError(f"{cls.__name__} doesn't take params")
        return cls(targets)

    def target_iter(self, n_qubits):
        return slicing(self.targets, n_qubits)


def slicing_singlevalue(arg: Union[slice, int], length: int) -> Iterator[int]:
    if isinstance(arg, slice):
        start, stop, step = arg.indices(length)
        i = start
        if step > 0:
            while i < stop:
                yield i
                i += step
        else:
            while i > stop:
                yield i
                i += step
    else:
        try:
            i = arg.__index__()
        except AttributeError:
            raise TypeError("indices must be integers or slices, not " + arg.__class__.__name__) from None
        if i < 0:
            i += length
        yield i


def slicing(args: Targets, length: int) -> Iterator[int]:
    if isinstance(args, tuple):
        for arg in args:
            yield from slicing_singlevalue(arg, length)
    else:
        yield from slicing_singlevalue(args, length)


def qubit_pairs(args: Tuple[Targets, Targets], length: int) -> Iterator[Tuple[int, int]]:
    if not isinstance(args, tuple) or len(args) != 2:
        raise ValueError("Control and target qubits pair(s) are required.")
    controls = list(slicing(args[0], length))
    targets = list(slicing(args[1], length))
    if len(controls) != len(targets):
        raise ValueError("The number of control qubits and target qubits must be the same.")
    for c, z in zip(controls, targets):
        if c == z:
            raise ValueError("Control qubit and target qubit must be different.")
    return zip(controls, targets)


def get_maximum_index(indices: Targets) -> int:
    def _maximum_idx_single(idx: int):
        if isinstance(idx, slice):
            start = -1
            stop = 0
            if idx.start is not None:
                start = idx.start.__index__()
            if idx.stop is not None:
                stop = idx.stop.__index__()
            return max(start, stop - 1)
        return idx.__index__()

    if isinstance(indices, tuple):
        return max((_maximum_idx_single(i) for i in indices), default=-1)
    return _maximum_idx_single(indices)


def find_n_qubits(gates: Iterable[Operation]) -> int:
    return max((get_maximum_index(g.targets) for g in gates), default=-1) + 1