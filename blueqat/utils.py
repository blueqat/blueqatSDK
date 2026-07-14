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
"""Integrated Quantum Operators, Utilities, VQE, and QAOA module with PyTorch.
Refactored and merged into a unified utils.py module with robust Autograd tracking.
"""

import cmath
import math
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from functools import reduce
from itertools import combinations, product
from math import pi
from numbers import Number, Integral
import typing
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import torch
import warnings

# 同層パッケージへの最小限の外部インポート
from .circuit import Circuit


# ==============================================================================
# SECTION 1: Pauli Operators & Algebra
# ==============================================================================

_PauliTuple = namedtuple("_PauliTuple", "n")
half_pi = pi / 2

_matrix: Dict[str, torch.Tensor] = {
    'I': torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
}

_mul_map: Dict[Tuple[str, str], Tuple[complex, str]] = {
    ('X', 'X'): (1.0, 'I'), ('X', 'Y'): (1j, 'Z'), ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'X'): (-1j, 'Z'), ('Y', 'Y'): (1.0, 'I'), ('Y', 'Z'): (1j, 'X'),
    ('Z', 'X'): (1j, 'Y'), ('Z', 'Y'): (-1j, 'X'), ('Z', 'Z'): (1.0, 'I'),
}

def _kron_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.kron(a, b)

def _kron_1d_rec(krons: list, lo: int, hi: int) -> torch.Tensor:
    if hi - lo == 1: return krons[lo]
    if hi - lo == 2: return _kron_1d(krons[lo], krons[lo + 1])
    mid = (lo + hi) // 2
    return _kron_1d(_kron_1d_rec(krons, lo, mid), _kron_1d_rec(krons, mid, hi))

def _term_to_dataarray(term: 'Term', n_qubits: int, device: torch.device) -> torch.Tensor:
    y_mat = torch.tensor([1j, -1j], dtype=torch.complex128, device=device)
    z_mat = torch.tensor([1, -1], dtype=torch.complex128, device=device)
    
    paulis = ['I'] * n_qubits
    for op in term.ops:
        paulis[op.n] = op.op
        
    data_list = []
    for g in paulis:
        if g == 'Y': data_list.append(y_mat.clone())
        elif g == 'Z': data_list.append(z_mat.clone())
        elif g in ('X', 'I'):
            data_list.append(torch.tensor([1, 1], dtype=torch.complex128, device=device))
            
    data_list.reverse()
    base_data = _kron_1d_rec(data_list, 0, len(data_list))
    return base_data * torch.as_tensor(term.coeff, dtype=torch.complex128, device=device)


class _PauliImpl:
    @property
    def op(self) -> str: return self.__class__.__name__[1]
    @property
    def is_identity(self) -> bool: return self.op == "I"
    @property
    def n_qubits(self) -> int: return 0 if self.is_identity else self.n + 1

    def __hash__(self) -> int: return hash((self.op, getattr(self, 'n', -1)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PauliImpl):
            if self.is_identity: return other.is_identity
            return self.n == getattr(other, 'n', None) and self.op == other.op
        if isinstance(other, Term): return self.to_term() == other
        if isinstance(other, Expr): return self.to_expr() == other
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)): return Term.from_pauli(self, other)
        if not isinstance(other, _PauliImpl): return NotImplemented
        if self.is_identity: return other.to_term()
        if other.is_identity: return self.to_term()
        if self.n == other.n and self.op == other.op: return I.to_term()
        return Term.from_paulipair(self, other)

    def __rmul__(self, other: Any) -> Any:
        return Term.from_pauli(self, other) if isinstance(other, (Number, torch.Tensor)) else NotImplemented

    def __truediv__(self, other: Any) -> Any:
        return Term.from_pauli(self, 1.0 / other) if isinstance(other, (Number, torch.Tensor)) else NotImplemented

    def __add__(self, other: Any) -> 'Expr': return self.to_expr() + other
    def __radd__(self, other: Any) -> 'Expr': return other + self.to_expr()
    def __sub__(self, other: Any) -> 'Expr': return self.to_expr() - other
    def __rsub__(self, other: Any) -> 'Expr': return other - self.to_expr()
    def __neg__(self) -> 'Term': return Term.from_pauli(self, -1.0)
    def __repr__(self) -> str: return "I" if self.is_identity else f"{self.op}[{self.n}]"

    def to_term(self) -> 'Term': return Term.from_pauli(self)
    def to_expr(self) -> 'Expr': return self.to_term().to_expr()
    
    @property
    def matrix(self) -> torch.Tensor: return _matrix[self.op].clone()
    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.to_term().to_matrix(n_qubits, sparse=sparse, device=device)


class _X(_PauliImpl, _PauliTuple): pass
class _Y(_PauliImpl, _PauliTuple): pass
class _Z(_PauliImpl, _PauliTuple): pass

class _PauliCtor:
    def __init__(self, ty: Type) -> None: self.ty = ty
    def __call__(self, n: int) -> _PauliImpl: return self.ty(n)
    def __getitem__(self, n: int) -> _PauliImpl: return self.ty(n)
    @property
    def matrix(self) -> torch.Tensor: return _matrix[self.ty.__name__[-1]].clone()

X = _PauliCtor(_X)
Y = _PauliCtor(_Y)
Z = _PauliCtor(_Z)

class _I(_PauliImpl, namedtuple("_I", "")):
    def __call__(self) -> '_I': return self
    @property
    def matrix(self) -> torch.Tensor: return _matrix['I'].clone()

I = _I()
_TermTuple = namedtuple("_TermTuple", "ops coeff")


class Term(_TermTuple):
    @staticmethod
    def from_paulipair(pauli1: Any, pauli2: Any) -> 'Term': return Term(Term.join_ops((pauli1, ), (pauli2, )), 1.0)
    @staticmethod
    def from_pauli(pauli: Any, coeff: Any = 1.0) -> 'Term': return Term((), coeff) if pauli.is_identity else Term((pauli, ), coeff)
    @staticmethod
    def from_ops_iter(ops: Any, coeff: Any) -> 'Term': return Term(tuple(ops), coeff)
    
    @staticmethod
    def from_chars(chars: Any) -> 'Term':
        paulis = [pauli_from_char(c, n) for n, c in enumerate(chars) if c != "I"]
        return 1.0 * I if not paulis else reduce(lambda a, b: a * b, paulis)

    @staticmethod
    def join_ops(ops1: tuple, ops2: tuple) -> tuple:
        i, j = len(ops1) - 1, 0
        while i >= 0 and j < len(ops2):
            if ops1[i] == ops2[j]: i, j = i - 1, j + 1
            else: break
        return ops1[:i + 1] + ops2[j:]

    @property
    def is_identity(self) -> bool: return not self.ops

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)): return Term(self.ops, self.coeff * other)
        if isinstance(other, Term): return Term(Term.join_ops(self.ops, other.ops), self.coeff * other.coeff)
        if isinstance(other, _PauliImpl): return self if other.is_identity else Term(Term.join_ops(self.ops, (other, )), self.coeff)
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)): return Term(self.ops, self.coeff * other)
        if isinstance(other, _PauliImpl): return self if other.is_identity else Term(Term.join_ops((other, ), self.ops), self.coeff)
        return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        return Term(self.ops, self.coeff / other) if isinstance(other, (Number, torch.Tensor)) else NotImplemented

    def __pow__(self, n: Any) -> Any:
        if isinstance(n, Integral):
            if n < 0: raise ValueError("n shall not be negative.")
            return Term.from_pauli(I) if n == 0 else Term(self.ops * n, self.coeff**n)
        return NotImplemented

    def __add__(self, other: Any) -> 'Expr': return Expr.from_term(self) + other
    def __radd__(self, other: Any) -> 'Expr': return other + Expr.from_term(self)
    def __sub__(self, other: Any) -> 'Expr': return Expr.from_term(self) - other
    def __rsub__(self, other: Any) -> 'Expr': return other - self.to_expr()
    def __neg__(self) -> 'Term': return Term(self.ops, -self.coeff)

    def __repr__(self) -> str:
        coeff_str = str(self.coeff.item()) if isinstance(self.coeff, torch.Tensor) else str(self.coeff)
        if not self.ops: return f"{coeff_str}*I"
        return f"{coeff_str}*" + "*".join(f"{op.op}[{op.n}]" for op in self.ops)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PauliImpl): other = other.to_term()
        if isinstance(other, Term): return _TermTuple.__eq__(self.simplify(), other.simplify())
        if isinstance(other, Expr): return NotImplemented  # let Expr.__eq__(other, self) handle it
        return False

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def to_term(self) -> 'Term': return self
    def to_expr(self) -> 'Expr': return Expr.from_term(self)

    def simplify(self) -> 'Term':
        def mul(op1: str, op2: str) -> Tuple[complex, str]:
            return (1.0, op2) if op1 == "I" else ((1.0, op1) if op2 == "I" else _mul_map[op1, op2])

        before = defaultdict(list)
        for op in self.ops:
            if op.op != "I": before[op.n].append(op.op)
            
        new_coeff = self.coeff
        new_ops = []
        for n in sorted(before.keys()):
            ops = before[n]
            k = 1.0
            op = ops[0]
            for _op in ops[1:]:
                _k, op = mul(op, _op)
                k *= _k
            new_coeff = new_coeff * k
            if isinstance(new_coeff, torch.Tensor):
                # .imag raises for non-complex dtypes, so only touch it when the
                # tensor is actually complex (e.g. a real theta*Z[0] coefficient
                # that never got promoted to complex must be left alone).
                if torch.is_complex(new_coeff) and new_coeff.imag == 0: new_coeff = new_coeff.real
            elif isinstance(new_coeff, complex) and new_coeff.imag == 0:
                new_coeff = new_coeff.real
            if op != "I": new_ops.append(pauli_from_char(op, n))
        return Term(tuple(new_ops), new_coeff)

    def n_iter(self) -> Iterator[int]: return (op.n for op in self.ops)
    def max_n(self) -> int:
        try: return max(self.n_iter())
        except ValueError: return -1
    @property
    def n_qubits(self) -> int: return self.max_n() + 1

    def is_commutable_with(self, other: Any) -> bool: return is_commutable(self, other)

    def get_time_evolution(self) -> Any:
        term = self.simplify()
        coeff, ops = term.coeff, term.ops

        def append_to_circuit(circuit: Any, t: float) -> None:
            if not ops: return
            for op in ops:
                if op.op == "X": circuit.h[op.n]
                elif op.op == "Y": circuit.rx(-half_pi)[op.n]
            for i in range(1, len(ops)):
                circuit.cx[ops[i - 1].n, ops[i].n]
            
            circuit.rz(-2 * coeff * t)[ops[-1].n]
            
            for i in range(len(ops) - 1, 0, -1):
                circuit.cx[ops[i - 1].n, ops[i].n]
            for op in ops:
                if op.op == "X": circuit.h[op.n]
                elif op.op == "Y": circuit.rx(half_pi)[op.n]

        return append_to_circuit

    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None: device = torch.device('cpu')
        if n_qubits == -1: n_qubits = self.n_qubits
        if n_qubits == 0:
            m = torch.as_tensor([[self.coeff]], dtype=torch.complex128, device=device)
            return m.to_sparse() if sparse else m

        dim = 2**n_qubits
        term = self.simplify()
        xor_bits = sum(1 << op.n for op in term.ops if op.op in ('X', 'Y'))
        
        cols = torch.arange(dim, dtype=torch.int64, device=device)
        rows = cols ^ xor_bits
        vals = _term_to_dataarray(term, n_qubits, device)
        
        if sparse:
            return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (dim, dim), dtype=torch.complex128, device=device)
        m = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
        m[rows, cols] = vals
        return m

_ExprTuple = namedtuple("_ExprTuple", "terms")


class Expr(_ExprTuple):
    @staticmethod
    def from_number(num: Any) -> 'Expr': return Expr.zero() if num == 0 else Expr.from_term(Term((), num))
    @staticmethod
    def from_term(term: Term) -> 'Expr': return Expr((term, ))
    @staticmethod
    def from_terms_iter(terms: Any) -> 'Expr': return Expr(tuple(term for term in terms))
    def terms_to_dict(self) -> dict:
        # Sum coefficients on collision rather than overwrite: a plain dict
        # comprehension would silently drop earlier terms if self.terms ever
        # contains duplicate `ops` keys (e.g. an Expr built via from_terms_iter
        # with unmerged input).
        d: dict = {}
        for op, coeff in self.terms:
            d[op] = d[op] + coeff if op in d else coeff
        return d
    @staticmethod
    def from_terms_dict(terms_dict: dict) -> 'Expr': return Expr(tuple(Term(k, v) for k, v in terms_dict.items()))
    @staticmethod
    def zero() -> 'Expr': return Expr(())

    @property
    def is_identity(self) -> bool:
        return True if not self.terms else (len(self.terms) == 1 and not self.terms[0].ops and self.terms[0].coeff == 1.0)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (_PauliImpl, Term)): other = other.to_expr()
        return self.simplify().terms == other.simplify().terms if isinstance(other, Expr) else False

    def __ne__(self, other: Any) -> bool: return not self.__eq__(other)

    def __add__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)): other = Expr.from_number(other)
        elif isinstance(other, Term): other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                terms[op] = terms[op] + coeff if op in terms else coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __sub__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)): other = Expr.from_number(other)
        elif isinstance(other, Term): other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                terms[op] = terms[op] - coeff if op in terms else -coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __radd__(self, other: Any) -> 'Expr': return Expr.from_number(other) + self if isinstance(other, (Number, torch.Tensor)) else NotImplemented
    def __rsub__(self, other: Any) -> 'Expr': return Expr.from_number(other) - self if isinstance(other, (Number, torch.Tensor)) else NotImplemented
    def __neg__(self) -> 'Expr': return Expr(tuple(Term(op, -coeff) for op, coeff in self.terms))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)): return Expr.from_terms_iter(Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl): other = other.to_term()
        if isinstance(other, Term): return Expr(tuple(term * other for term in self.terms))
        if isinstance(other, Expr):
            terms = defaultdict(float)
            for t1, t2 in product(self.terms, other.terms):
                term = t1 * t2
                terms[term.ops] = terms[term.ops] + term.coeff if term.ops in terms else term.coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)): return Expr.from_terms_iter(Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl): other = other.to_term()
        return Expr(tuple(other * term for term in self.terms)) if isinstance(other, Term) else NotImplemented

    def __truediv__(self, other: Any) -> Any: return Expr(tuple(term / other for term in self.terms)) if isinstance(other, (Number, torch.Tensor)) else NotImplemented
    def __iter__(self) -> Iterator[Term]: return iter(self.terms)
    def __getnewargs__(self) -> Tuple[Tuple[Term, ...]]: return (self.terms, )
    def __repr__(self) -> str: return "0*I" if not self.terms else " + ".join(repr(term) for term in self.terms)
    def to_expr(self) -> 'Expr': return self
    def max_n(self) -> int:
        try: return max(term.max_n() for term in self.terms if term.ops)
        except ValueError: return -1
    def is_commutable_with(self, other: Any) -> bool: return is_commutable(self, other)
    def is_all_terms_commutable(self) -> bool:
        return all(is_commutable(a, b) for a, b in combinations(self.terms, 2))
    @property
    def n_qubits(self) -> int: return self.max_n() + 1
    def coeffs(self) -> Iterator[Any]:
        for term in self.terms: yield term.coeff

    def simplify(self) -> 'Expr':
        d = defaultdict(float)
        for term in self.terms:
            term = term.simplify()
            d[term.ops] = d[term.ops] + term.coeff if term.ops in d else term.coeff
        return Expr.from_terms_iter(Term.from_ops_iter(k, d[k]) for k in sorted(d, key=repr) if d[k])

    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None: device = torch.device('cpu')
        if n_qubits == -1: n_qubits = self.n_qubits
        dim = 2**n_qubits
        expr = self.simplify()
        
        if sparse:
            total_matrix = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.int64, device=device), torch.empty(0, dtype=torch.complex128, device=device), (dim, dim))
            for term in expr.terms: total_matrix = total_matrix + term.to_matrix(n_qubits, sparse=True, device=device)
            return total_matrix.coalesce()
        else:
            total_matrix = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
            for term in expr.terms: total_matrix = total_matrix + term.to_matrix(n_qubits, sparse=False, device=device)
            return total_matrix

def pauli_from_char(ch: str, n: int = 0) -> '_PauliImpl':
    ch = ch.upper()
    if ch == "I": return I
    if ch == "X": return X(n)
    if ch == "Y": return Y(n)
    if ch == "Z": return Z(n)
    raise ValueError("ch shall be X, Y, Z or I")

def term_from_chars(chars: str) -> 'Term':
    """Make Pauli's Term from chars written as 'X', 'Y', 'Z' or 'I'."""
    return Term.from_chars(reversed(chars))

def commutator(expr1: Any, expr2: Any) -> 'Expr':
    """Returns [expr1, expr2] = expr1 * expr2 - expr2 * expr1."""
    expr1 = expr1.to_expr().simplify()
    expr2 = expr2.to_expr().simplify()
    return (expr1 * expr2 - expr2 * expr1).simplify()

def is_commutable(expr1: Any, expr2: Any, eps: float = 1e-8) -> bool:
    """Test whether expr1 and expr2 are commutable."""
    return sum((x * x.conjugate()).real for x in commutator(expr1, expr2).coeffs()) < eps

def qubo_bit(n: int) -> Expr:
    return 0.5 - 0.5 * Z[n]

def from_qubo(qubo: Sequence[Sequence[float]]) -> Expr:
    h = 0.0
    for i in range(len(qubo)):
        h += qubo_bit(i) * qubo[i][i]
        for j in range(i + 1, len(qubo)):
            h += qubo_bit(i) * qubo_bit(j) * (qubo[i][j] + qubo[j][i])
    return h


# ==============================================================================
# SECTION 2: General Utility Formats
# ==============================================================================

def to_inttuple(bitstr: Union[str, Counter, Dict[str, int]]) -> Union[Tuple[int, ...], Counter, Dict[Tuple[int, ...], int]]:
    if isinstance(bitstr, str): return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter): return Counter({tuple(int(b) for b in k): v for k, v in bitstr.items()})
    if isinstance(bitstr, dict): return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")

def ignore_global_phase(statevec: torch.Tensor) -> torch.Tensor:
    """Multiply e^-iθ to `statevec` where θ is a phase of first non-zero element."""
    for q in statevec:
        if torch.abs(q) > 1e-7:
            ang = torch.abs(q) / q
            statevec = statevec * ang
            break
    return statevec

def gen_graycode(n: int) -> Iterator[int]:
    return (v ^ (v >> 1) for v in range(2**n))


def gen_gray_controls(n: int) -> Iterator[Tuple[int, int, int]]:
    """Generate an iterator which returns bit indices for constructing
    Gray code based controlled gate.
    """
    def gen_changedbit(n_bits: int) -> Iterator[int]:
        pow2 = [2 ** i for i in range(n_bits)]
        gen = gen_graycode(n_bits)
        try:
            prev = next(gen)
        except StopIteration:
            raise ValueError("Empty Gray code generation.") from None
        for g in gen:
            yield pow2.index(g ^ prev)
            prev = g

    def gen_cxtarget() -> Iterator[int]:
        k = 0
        while True:
            for _ in range(2**k):
                yield k
            k += 1

    def gen_parity() -> Iterator[int]:
        while True:
            yield 0
            yield 1

    for c0, c1, p in zip(gen_changedbit(n), gen_cxtarget(), gen_parity()):
        if c0 == c1:
            yield c0 - 1, c1, p
        else:
            yield c0, c1, p


def check_unitarity(mat: torch.Tensor) -> bool:
    """Check whether mat is a unitary matrix."""
    if mat.dim() != 2 or mat.shape[0] != mat.shape[1]:
        return False
    eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
    return torch.allclose(mat @ mat.mH, eye, atol=1e-6)


def calc_u_params(mat: torch.Tensor) -> Tuple[float, float, float, float]:
    """Calculate U-gate parameters from a 2x2 unitary matrix."""
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    gamma = cmath.phase(complex(mat[0, 0]))
    phase = cmath.exp(-1j * gamma)
    m00 = complex(mat[0, 0]) * phase
    m10 = complex(mat[1, 0]) * phase
    m11 = complex(mat[1, 1]) * phase
    theta = math.atan2(abs(m10), m00.real) * 2.0
    phi_plus_lambda = cmath.phase(m11)
    phi = cmath.phase(m10) % (2.0 * math.pi)
    lam = (phi_plus_lambda - phi) % (2.0 * math.pi)
    return theta, phi, lam, gamma


def sqrt_2x2_matrix(mat: torch.Tensor) -> torch.Tensor:
    """Returns square root of a 2x2 matrix.

    Reference: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    assert mat.shape == (2, 2)
    eye = torch.eye(2, dtype=mat.dtype, device=mat.device)
    s = torch.sqrt(torch.linalg.det(mat))
    t = torch.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    if abs(complex(t)) < 1e-8:  # Avoid division by zero
        s = -s
        t = torch.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return (mat + s * eye) / t


# ==============================================================================
# SECTION 3: VQE / QAOA Ansatz Execution Framework
# ==============================================================================

class AnsatzBase:
    """Base class for Variational Quantum Eigensolver Ansatz using PyTorch."""
    def __init__(self, hamiltonian: Any, n_params: int) -> None:
        self.hamiltonian = hamiltonian
        self.n_params = n_params
        self.n_qubits: int = self.hamiltonian.max_n() + 1
        self.sparse: Optional[torch.Tensor] = None

    def make_sparse(self, sparse: bool = True, device: Optional[torch.device] = None) -> None:
        # self.n_qubits may be wider than the hamiltonian's own qubit span (e.g. an
        # init_circuit with extra/ancilla qubits), so it must be passed explicitly --
        # otherwise to_matrix() infers a narrower width and later matrix-vector ops
        # against the full-width statevector fail with a dimension mismatch.
        self.sparse = self.hamiltonian.to_matrix(self.n_qubits, sparse=sparse, device=device)

    def get_circuit(self, params: torch.Tensor) -> Circuit: raise NotImplementedError

    def get_energy(self, circuit: Circuit, sampler: Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]) -> torch.Tensor:
        """Calculate energy expectation value from circuit and sampler with Autograd support.

        Whether the result carries a gradient back to `circuit`'s parameters depends
        on `sampler`: an exact sampler (e.g. `non_sampling_sampler`) keeps the
        autograd graph intact, while a genuinely stochastic one (e.g. one built from
        `get_measurement_sampler`) does not -- real shot noise isn't differentiable,
        so that is expected, not a bug.
        """
        val: Any = 0.0

        for raw_meas in self.hamiltonian:
            # Merge any operators sharing a qubit (e.g. X[0]*Z[0] -> -1j*Y[0]) into a
            # single effective Pauli per qubit first. Without this, a term touching
            # the same qubit more than once would get an extra basis rotation applied
            # to it and would have that qubit's bit counted more than once in the
            # parity check below, corrupting the sign of the contribution.
            meas = raw_meas.simplify()
            coeff_val = meas.coeff if isinstance(meas.coeff, torch.Tensor) else complex(meas.coeff)

            # 1. 定数項（Iのみ）の処理
            if not meas.ops:
                val = val + coeff_val
                continue

            # 2. この項に関係する全ての量子ビットを特定
            active_qubits = sorted(op.n for op in meas.ops)
            n_qubits = max(max(active_qubits) + 1, circuit.n_qubits)

            # 3. 各項ごとに完全に独立した測定用回路を作成
            c = Circuit(n_qubits)
            c.ops = list(circuit.ops)

            for op in meas.ops:
                if op.op == "X":
                    c.h[op.n]
                elif op.op == "Y":
                    c.rx(torch.tensor(-torch.pi / 2, dtype=torch.float64))[op.n]

            # 4. サンプラーが返す (測定qubitごとのbit tuple -> 確率) を実際に消費し、
            #    各qubitの1ビットのパリティで符号を決めて集計する
            #    (simplify() 済みなので active_qubits は重複なく meas.ops と1対1)
            for bits, prob in sampler(c, active_qubits).items():
                parity = sum(bits) % 2
                val = val + (-prob * coeff_val if parity else prob * coeff_val)

        if isinstance(val, torch.Tensor):
            return (val.real if torch.is_complex(val) else val).squeeze()
        return torch.tensor(val.real if isinstance(val, complex) else val, dtype=torch.float64)

    def get_energy_sparse(self, circuit: Circuit) -> torch.Tensor:
        return sparse_expectation(self.sparse, circuit.run())

    def get_objective(self, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None, 
                      device: Optional[torch.device] = None) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.sparse is None: self.make_sparse(sparse=True, device=device)
        if sampler is not None: return lambda p: self.get_energy(self.get_circuit(p), sampler)
        return lambda p: self.get_energy_sparse(self.get_circuit(p))


class QaoaAnsatz(AnsatzBase):
    def __init__(self, hamiltonian: Any, step: int = 1, init_circuit: Optional[Circuit] = None, mixer: Optional[Any] = None) -> None:
        # Convert to Expr before super().__init__, which immediately calls
        # .max_n() on it -- a bare Pauli operator like Z[0] (as opposed to a
        # Term/Expr) doesn't have that method and would raise AttributeError.
        hamiltonian = hamiltonian.to_expr().simplify()
        super().__init__(hamiltonian, step * 2)
        self.hamiltonian = hamiltonian
        if not self.check_hamiltonian():
            raise ValueError("Hamiltonian terms are not commutable")
        self.step = step
        self.n_qubits = self.hamiltonian.max_n() + 1
        
        if init_circuit:
            self.init_circuit = init_circuit
            if init_circuit.n_qubits > self.n_qubits: self.n_qubits = init_circuit.n_qubits
        else:
            if mixer: raise ValueError('init_circuit is required when mixer is not default.')
            self.init_circuit = Circuit(self.n_qubits).h[:]
            
        self.mixer = mixer
        self.time_evolutions = [term.get_time_evolution() for term in self.hamiltonian]
        self.mixer_time_evolutions = [term.get_time_evolution() for term in self.mixer] if mixer else []

    def check_hamiltonian(self) -> bool:
        """Check hamiltonian is commutable. This condition is required for QaoaAnsatz,
        since get_circuit Trotterizes e^{-iHt} into a per-term product of time
        evolutions -- exact only when every term commutes with every other term."""
        return self.hamiltonian.is_all_terms_commutable()

    def get_circuit(self, params: torch.Tensor) -> Circuit:
        c = self.init_circuit.copy()
        betas, gammas = params[:self.step], params[self.step:]
        for beta, gamma in zip(betas, gammas):
            for evo in self.time_evolutions: evo(c, gamma * 2.0 * torch.pi)
            if self.mixer is None: c.rx(beta * torch.pi)[:]
            else:
                for evo in self.mixer_time_evolutions: evo(c, beta * torch.pi)
        return c


@dataclass
class VqeResult:
    vqe: Optional['Vqe'] = None
    params: Optional[torch.Tensor] = None
    circuit: Optional[Circuit] = None
    _probs: Optional[Dict[Tuple[int, ...], float]] = None

    def most_common(self, n: int = 1) -> Tuple[Tuple[Tuple[int, ...], float], ...]:
        return tuple(sorted(self.get_probs().items(), key=lambda item: -item[1]))[:n]

    def get_probs(self, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None, 
                  rerun: Optional[bool] = None, store: bool = True) -> Dict[Tuple[int, ...], float]:
        if rerun is None: rerun = sampler is not None
        if self._probs is not None and not rerun: return self._probs
        if sampler is None and self.vqe is not None: sampler = self.vqe.sampler
        if self.circuit is None: raise ValueError("No circuit available.")

        raw_probs = expect(self.circuit.run(), range(self.circuit.n_qubits)) if sampler is None else sampler(self.circuit, range(self.circuit.n_qubits))
        # get_probs()/most_common() are reporting APIs (sorting, printing, equality
        # checks against plain dicts), not part of an autograd graph, so normalize
        # to plain floats regardless of whether expect() or a custom sampler handed
        # back tensors.
        probs = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in raw_probs.items()}
        if store: self._probs = probs
        return probs


class Vqe:
    def __init__(self, ansatz: AnsatzBase, optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None, sampler: Optional[Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]] = None) -> None:
        self.ansatz = ansatz
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.05}
        self.sampler = sampler
        self.sampler_call_count = 0

    def run(self, max_iter: int = 500, tol: float = 1e-6, verbose: bool = False, device: Optional[torch.device] = None,
            initial_params: Optional[torch.Tensor] = None) -> VqeResult:
        if device is None: device = torch.device('cpu')
        self.sampler_call_count = 0
        counting_sampler = None
        if self.sampler is not None:
            def counting_sampler(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
                self.sampler_call_count += 1
                return self.sampler(circuit, meas)
        objective_fn = self.ansatz.get_objective(counting_sampler, device=device)

        if initial_params is None:
            params = torch.rand(self.ansatz.n_params, dtype=torch.float64, device=device, requires_grad=True)
        else:
            params = torch.as_tensor(initial_params, dtype=torch.float64, device=device).clone().detach().requires_grad_(True)
            if params.shape != (self.ansatz.n_params,):
                raise ValueError(f"initial_params must have shape ({self.ansatz.n_params},), got {tuple(params.shape)}")
        optimizer = self.optimizer_cls([params], **self.optimizer_kwargs)
        
        for idx in range(max_iter):
            optimizer.zero_grad()
            loss = objective_fn(params)
            loss.backward()
            optimizer.step()
            if params.grad is not None and torch.norm(params.grad) < tol: break
                
        final_params = params.detach()
        self._result = VqeResult(self, final_params, self.ansatz.get_circuit(final_params))
        return self._result


def expect(qubits: torch.Tensor, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], torch.Tensor]:
    """Marginal probabilities of `meas` qubits, as gradient-carrying tensors (not
    plain floats) so that `AnsatzBase.get_energy` can backprop through them when
    `qubits` came from a differentiable circuit run."""
    meas_tuple = tuple(meas)
    mask = reduce(lambda acc, v: acc | (1 << v), meas_tuple, 0)
    cnt: Dict[int, torch.Tensor] = {}
    probs = torch.abs(qubits) ** 2

    for i, p_val in enumerate(probs):
        # .item() here is only a control-flow check (skip exactly-zero-probability
        # outcomes, matching the previous behavior); p_val itself -- what actually
        # gets accumulated -- stays a tensor so the gradient is preserved.
        if p_val.item() == 0.0: continue
        key = i & mask
        cnt[key] = cnt[key] + p_val if key in cnt else p_val
    return {tuple(1 if k & (1 << i) else 0 for i in meas_tuple): val for k, val in cnt.items()}

def non_sampling_sampler(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
    return expect(circuit.run(), meas)

def get_measurement_sampler(n_sample: int, device: Optional[torch.device] = None) -> Callable[[Circuit, typing.Iterable[int]], Dict[Tuple[int, ...], float]]:
    def sampling_by_measurement(circuit: Circuit, meas: typing.Iterable[int]) -> Dict[Tuple[int, ...], float]:
        meas_tuple = tuple(meas)
        statevector = circuit.run()
        probs = torch.abs(statevector) ** 2
        
        samples = torch.multinomial(probs, n_sample, replacement=True)
        unique_elements, counts = torch.unique(samples, return_counts=True)
        
        result_counts = Counter()
        for idx, count in zip(unique_elements, counts):
            bit_key = tuple((idx.item() >> m) & 1 for m in meas_tuple)
            result_counts[bit_key] += count.item()
        return {k: v / n_sample for k, v in result_counts.items()}
    return sampling_by_measurement

def sparse_expectation(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    mv = torch.sparse.mm(mat, vec.unsqueeze(1)).squeeze(1) if mat.is_sparse else torch.mv(mat, vec)
    return torch.vdot(vec, mv).real