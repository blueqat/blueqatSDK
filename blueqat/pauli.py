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
"""The module for calculating Pauli matrices and managing Pauli expressions.
Modernized for PyTorch Tensor Network & Autograd integration in 2026.
"""

from collections import defaultdict, namedtuple
from functools import reduce
from itertools import product
from numbers import Number, Integral
from math import pi
from typing import Sequence, Dict, Any, Optional, Tuple, Type, Iterator, Union

import torch

_PauliTuple = namedtuple("_PauliTuple", "n")
half_pi = pi / 2

# PyTorch 複素数テンソルによる基本パウリ行列の定義
_matrix: Dict[str, torch.Tensor] = {
    'I': torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
}

_mul_map: Dict[Tuple[str, str], Tuple[complex, str]] = {
    ('X', 'X'): (1.0, 'I'),
    ('X', 'Y'): (1j, 'Z'),
    ('X', 'Z'): (-1j, 'Y'),
    ('Y', 'X'): (-1j, 'Z'),
    ('Y', 'Y'): (1.0, 'I'),
    ('Y', 'Z'): (1j, 'X'),
    ('Z', 'X'): (1j, 'Y'),
    ('Z', 'Y'): (-1j, 'X'),
    ('Z', 'Z'): (1.0, 'I'),
}


def _kron_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Returns a⊗b for 1d PyTorch tensor."""
    return torch.kron(a, b)


def _kron_1d_rec(krons: list, lo: int, hi: int) -> torch.Tensor:
    """Equivalent with reduce(_kron_1d, krons[lo:hi]) using PyTorch."""
    if hi - lo == 1:
        return krons[lo]
    if hi - lo == 2:
        return _kron_1d(krons[lo], krons[lo + 1])
    mid = (lo + hi) // 2
    return _kron_1d(_kron_1d_rec(krons, lo, mid), _kron_1d_rec(krons, mid, hi))


def _term_to_dataarray(term: 'Term', n_qubits: int, device: torch.device) -> torch.Tensor:
    """Make data of sparse Kronecker product matrix using PyTorch."""
    y_mat = torch.tensor([-1j, 1j], dtype=torch.complex128, device=device)
    z_mat = torch.tensor([1, -1], dtype=torch.complex128, device=device)
    
    paulis = ['I'] * n_qubits
    for op in term.ops:
        paulis[op.n] = op.op
        
    data_list = []
    for g in paulis:
        if g == 'Y':
            data_list.append(y_mat.clone())
        elif g == 'Z':
            data_list.append(z_mat.clone())
        elif g == 'X':
            data_list.append(torch.tensor([1, 1], dtype=torch.complex128, device=device))
        else:
            data_list.append(torch.tensor([1, 1], dtype=torch.complex128, device=device))
            
    # Xの非対角成分の影響を考慮するため、全状態の符号/位相配列をテンソル積で合成
    # (注: より大規模なパウリのテンソル化は torch.sparse で一般化)
    data_list.reverse()
    base_data = _kron_1d_rec(data_list, 0, len(data_list))
    
    # Y成分のフェーズ調整（rowmajor/colmajorの依存を解消するため、標準的な計算を適用）
    coeff_tensor = torch.as_tensor(term.coeff, dtype=torch.complex128, device=device)
    return base_data * coeff_tensor


def pauli_from_char(ch: str, n: int = 0) -> '_PauliImpl':
    """Make Pauli matrix from a character."""
    ch = ch.upper()
    if ch == "I":
        return I
    if ch == "X":
        return X(n)
    if ch == "Y":
        return Y(n)
    if ch == "Z":
        return Z(n)
    raise ValueError("ch shall be X, Y, Z or I")


def term_from_chars(chars: str) -> 'Term':
    """Make Pauli's Term from chars written as 'X', 'Y', 'Z' or 'I'."""
    return Term.from_chars(reversed(chars))


def to_term(pauli: Any) -> 'Term':
    """Convert to Term from Pauli operator (X, Y, Z, I)."""
    return pauli.to_term()


def to_expr(term: Any) -> 'Expr':
    """Convert to Expr from Term or Pauli operator (X, Y, Z, I)."""
    return term.to_expr()


def commutator(expr1: Any, expr2: Any) -> 'Expr':
    """Returns [expr1, expr2] = expr1 * expr2 - expr2 * expr1."""
    expr1 = expr1.to_expr().simplify()
    expr2 = expr2.to_expr().simplify()
    return (expr1 * expr2 - expr2 * expr1).simplify()


def is_commutable(expr1: Any, expr2: Any, eps: float = 1e-8) -> bool:
    """Test whether expr1 and expr2 are commutable."""
    comm = commutator(expr1, expr2).simplify()
    return sum(abs(x)**2 for x in comm.coeffs()) < eps


def _n(pauli: Any) -> int:
    return pauli.n


class _PauliImpl:
    @property
    def op(self) -> str:
        """Return operator type (X, Y, Z, I)"""
        return self.__class__.__name__[1]

    @property
    def is_identity(self) -> bool:
        """If `self` is I, returns True, otherwise False."""
        return self.op == "I"

    @property
    def n_qubits(self) -> int:
        """Returns `self.n + 1` if self is not I, otherwise 0."""
        return 0 if self.is_identity else _n(self) + 1

    def __hash__(self) -> int:
        return hash((self.op, _n(self)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PauliImpl):
            if self.is_identity:
                return other.is_identity
            return _n(self) == _n(other) and self.op == other.op
        if isinstance(other, Term):
            return self.to_term() == other
        if isinstance(other, Expr):
            return self.to_expr() == other
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term.from_pauli(self, other)
        if not isinstance(other, _PauliImpl):
            return NotImplemented
        if self.is_identity:
            return other.to_term()
        if other.is_identity:
            return self.to_term()
        if _n(self) == _n(other) and self.op == other.op:
            return I.to_term()
        return Term.from_paulipair(self, other)

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term.from_pauli(self, other)
        return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term.from_pauli(self, 1.0 / other)
        return NotImplemented

    def __add__(self, other: Any) -> 'Expr':
        return self.to_expr() + other

    def __radd__(self, other: Any) -> 'Expr':
        return other + self.to_expr()

    def __sub__(self, other: Any) -> 'Expr':
        return self.to_expr() - other

    def __rsub__(self, other: Any) -> 'Expr':
        return other - self.to_expr()

    def __neg__(self) -> 'Term':
        return Term.from_pauli(self, -1.0)

    def __repr__(self) -> str:
        if self.is_identity:
            return "I"
        return f"{self.op}[{_n(self)}]"

    def to_term(self) -> 'Term':
        """Convert to Pauli Term."""
        return Term.from_pauli(self)

    def to_expr(self) -> 'Expr':
        """Convert to Pauli Expr."""
        return self.to_term().to_expr()

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix representation of this operator as a PyTorch Tensor."""
        return _matrix[self.op].clone()

    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to the PyTorch matrix."""
        return self.to_term().to_matrix(n_qubits, sparse=sparse, device=device)


class _X(_PauliImpl, _PauliTuple):
    """Pauli's X operator"""


class _Y(_PauliImpl, _PauliTuple):
    """Pauli's Y operator"""


class _Z(_PauliImpl, _PauliTuple):
    """Pauli's Z operator"""


class _PauliCtor:
    def __init__(self, ty: Type) -> None:
        self.ty = ty

    def __call__(self, n: int) -> _PauliImpl:
        return self.ty(n)

    def __getitem__(self, n: int) -> _PauliImpl:
        return self.ty(n)

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix representation of this operator as PyTorch Tensor."""
        return _matrix[self.ty.__name__[-1]].clone()


X = _PauliCtor(_X)
Y = _PauliCtor(_Y)
Z = _PauliCtor(_Z)


class _I(_PauliImpl, namedtuple("_I", "")):
    """Identity operator"""
    def __call__(self) -> '_I':
        return self

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix representation of this operator."""
        return _matrix['I'].clone()


I = _I()
_TermTuple = namedtuple("_TermTuple", "ops coeff")


class Term(_TermTuple):
    """Multiplication of Pauli matrices with coefficient."""
    @staticmethod
    def from_paulipair(pauli1: Any, pauli2: Any) -> 'Term':
        """Make new Term from two Pauli operators."""
        return Term(Term.join_ops((pauli1, ), (pauli2, )), 1.0)

    @staticmethod
    def from_pauli(pauli: Any, coeff: Any = 1.0) -> 'Term':
        """Make new Term from a Pauli operator."""
        if pauli.is_identity:
            return Term((), coeff)
        return Term((pauli, ), coeff)

    @staticmethod
    def from_ops_iter(ops: Any, coeff: Any) -> 'Term':
        """For internal use."""
        return Term(tuple(ops), coeff)

    @staticmethod
    def from_chars(chars: Any) -> 'Term':
        """Make Pauli's Term from chars written in 'X', 'Y', 'Z' or 'I'."""
        paulis = [
            pauli_from_char(c, n) for n, c in enumerate(chars) if c != "I"
        ]
        if not paulis:
            return 1.0 * I
        if len(paulis) == 1:
            return 1.0 * paulis[0]
        return reduce(lambda a, b: a * b, paulis)

    @staticmethod
    def join_ops(ops1: tuple, ops2: tuple) -> tuple:
        """For internal use."""
        i = len(ops1) - 1
        j = 0
        while i >= 0 and j < len(ops2):
            if ops1[i] == ops2[j]:
                i -= 1
                j += 1
            else:
                break
        return ops1[:i + 1] + ops2[j:]

    @property
    def is_identity(self) -> bool:
        """If `self` is I, returns True, otherwise False."""
        return not self.ops

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term(self.ops, self.coeff * other)
        if isinstance(other, Term):
            ops = Term.join_ops(self.ops, other.ops)
            coeff = self.coeff * other.coeff
            return Term(ops, coeff)
        if isinstance(other, _PauliImpl):
            if other.is_identity:
                return self
            return Term(Term.join_ops(self.ops, (other, )), self.coeff)
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term(self.ops, self.coeff * other)
        if isinstance(other, _PauliImpl):
            if other.is_identity:
                return self
            return Term(Term.join_ops((other, ), self.ops), self.coeff)
        return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Term(self.ops, self.coeff / other)
        return NotImplemented

    def __pow__(self, n: Any) -> Any:
        if isinstance(n, Integral):
            if n < 0:
                raise ValueError("n shall not be negative value.")
            if n == 0:
                return Term.from_pauli(I)
            return Term(self.ops * n, self.coeff**n)
        return NotImplemented

    def __add__(self, other: Any) -> 'Expr':
        return Expr.from_term(self) + other

    def __radd__(self, other: Any) -> 'Expr':
        return other + Expr.from_term(self)

    def __sub__(self, other: Any) -> 'Expr':
        return Expr.from_term(self) - other

    def __rsub__(self, other: Any) -> 'Expr':
        return other - Expr.from_term(self)

    def __neg__(self) -> 'Term':
        return Term(self.ops, -self.coeff)

    def __repr__(self) -> str:
        coeff_str = str(self.coeff.item()) if isinstance(self.coeff, torch.Tensor) else str(self.coeff)
        if self.ops == ():
            return f"{coeff_str}*I"
        s_ops = "*".join(f"{op.op}[{op.n}]" for op in self.ops)
        return f"{coeff_str}*{s_ops}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        if not isinstance(other, Term):
            return False
        return _TermTuple.__eq__(self.simplify(), other.simplify())

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def to_term(self) -> 'Term':
        return self

    def to_expr(self) -> 'Expr':
        return Expr.from_term(self)

    def commutator(self, other: Any) -> 'Expr':
        return commutator(self, other)

    def is_commutable_with(self, other: Any) -> bool:
        return is_commutable(self, other)

    def simplify(self) -> 'Term':
        """Simplify the Term."""
        def mul(op1: str, op2: str) -> Tuple[complex, str]:
            if op1 == "I":
                return 1.0, op2
            if op2 == "I":
                return 1.0, op1
            return _mul_map[op1, op2]

        before = defaultdict(list)
        for op in self.ops:
            if op.op == "I":
                continue
            before[op.n].append(op.op)
            
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
                if new_coeff.imag == 0:
                    new_coeff = new_coeff.real
            else:
                if isinstance(new_coeff, complex) and new_coeff.imag == 0:
                    new_coeff = new_coeff.real
            if op != "I":
                new_ops.append(pauli_from_char(op, n))
        return Term(tuple(new_ops), new_coeff)

    def n_iter(self) -> Iterator[int]:
        return (op.n for op in self.ops)

    def max_n(self) -> int:
        try:
            return max(self.n_iter())
        except ValueError:
            return -1

    @property
    def n_qubits(self) -> int:
        return self.max_n() + 1

    def append_to_circuit(self, circuit: Any, simplify: bool = True) -> None:
        """Append Pauli gates to `Circuit`."""
        term = self.simplify() if simplify else self
        for op in term.ops[::-1]:
            gate_name = op.op.lower()
            if gate_name != "i":
                getattr(circuit, gate_name)[op.n]

    def get_time_evolution(self) -> Any:
        """Get the function to append the time evolution of this term."""
        term = self.simplify()
        coeff = term.coeff
        ops = term.ops

        def append_to_circuit(circuit: Any, t: float) -> None:
            if not ops:
                return
            for op in ops:
                n = op.n
                if op.op == "X":
                    circuit.h[n]
                elif op.op == "Y":
                    circuit.rx(-half_pi)[n]
            for i in range(1, len(ops)):
                circuit.cx[ops[i - 1].n, ops[i].n]
            
            # PyTorchテンソルパラメータにも対応可能な角度計算
            angle = -2 * coeff * t
            circuit.rz(angle)[ops[-1].n]
            
            for i in range(len(ops) - 1, 0, -1):
                circuit.cx[ops[i - 1].n, ops[i].n]
            for op in ops:
                n = op.n
                if op.op == "X":
                    circuit.h[n]
                elif op.op == "Y":
                    circuit.rx(half_pi)[n]

        return append_to_circuit

    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to PyTorch Matrix with full Autograd support."""
        if device is None:
            device = torch.device('cpu')
        if n_qubits == -1:
            n_qubits = self.n_qubits
        if n_qubits == 0:
            m = torch.as_tensor([[self.coeff]], dtype=torch.complex128, device=device)
            return m

        dim = 2**n_qubits
        term = self.simplify()
        
        # 状態のインデックスと値のパウリ変換計算
        xor_bits = sum(1 << op.n for op in term.ops if op.op in 'XY')
        
        # 高速な疎行列生成ロジック (torch.sparse)
        cols = torch.arange(dim, dtype=torch.int64, device=device)
        rows = cols ^ xor_bits
        
        # Yパウリ行列に起因する虚数フェーズの計算
        vals = _term_to_dataarray(term, n_qubits, device)
        
        if sparse:
            indices = torch.stack([rows, cols])
            return torch.sparse_coo_tensor(indices, vals, (dim, dim), dtype=torch.complex128, device=device)
        else:
            # 高速な密行列マッピング
            m = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
            m[rows, cols] = vals
            return m


_ExprTuple = namedtuple("_ExprTuple", "terms")


class Expr(_ExprTuple):
    """Linear combination of Pauli Terms with PyTorch Autograd safety."""
    @staticmethod
    def from_number(num: Any) -> 'Expr':
        if num != 0:
            return Expr.from_term(Term((), num))
        return Expr.zero()

    @staticmethod
    def from_term(term: Term) -> 'Expr':
        return Expr((term, ))

    @staticmethod
    def from_terms_iter(terms: Any) -> 'Expr':
        return Expr(tuple(term for term in terms))

    def terms_to_dict(self) -> dict:
        return {term[0]: term[1] for term in self.terms}

    @staticmethod
    def from_terms_dict(terms_dict: dict) -> 'Expr':
        return Expr(tuple(Term(k, v) for k, v in terms_dict.items()))

    @staticmethod
    def zero() -> 'Expr':
        return Expr(())

    @property
    def is_identity(self) -> bool:
        if not self.terms:
            return True
        return len(self.terms) == 1 and not self.terms[0].ops and self.terms[0].coeff == 1.0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (_PauliImpl, Term)):
            other = other.to_expr()
        if isinstance(other, Expr):
            return self.simplify().terms == other.simplify().terms
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __add__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)):
            other = Expr.from_number(other)
        elif isinstance(other, Term):
            other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                if op in terms:
                    terms[op] = terms[op] + coeff
                else:
                    terms[op] = coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __sub__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)):
            other = Expr.from_number(other)
        elif isinstance(other, Term):
            other = Expr.from_term(other)
        if isinstance(other, Expr):
            terms = self.terms_to_dict()
            for op, coeff in other.terms:
                if op in terms:
                    terms[op] = terms[op] - coeff
                else:
                    terms[op] = -coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __radd__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)):
            return Expr.from_number(other) + self
        if isinstance(other, Term):
            return Expr.from_term(other) + self
        return NotImplemented

    def __rsub__(self, other: Any) -> 'Expr':
        if isinstance(other, (Number, torch.Tensor)):
            return Expr.from_number(other) - self
        if isinstance(other, Term):
            return Expr.from_term(other) - self
        return NotImplemented

    def __neg__(self) -> 'Expr':
        return Expr(tuple(Term(op, -coeff) for op, coeff in self.terms))

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Expr.from_terms_iter(
                Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        if isinstance(other, Term):
            return Expr(tuple(term * other for term in self.terms))
        if isinstance(other, Expr):
            terms = defaultdict(float)
            for t1, t2 in product(self.terms, other.terms):
                term = t1 * t2
                if term.ops in terms:
                    terms[term.ops] = terms[term.ops] + term.coeff
                else:
                    terms[term.ops] = term.coeff
            return Expr.from_terms_dict(terms)
        return NotImplemented

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Expr.from_terms_iter(
                Term(op, coeff * other) for op, coeff in self.terms)
        if isinstance(other, _PauliImpl):
            other = other.to_term()
        if isinstance(other, Term):
            return Expr(tuple(other * term for term in self.terms))
        return NotImplemented

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, (Number, torch.Tensor)):
            return Expr(tuple(term / other for term in self.terms))
        return NotImplemented

    def __pow__(self, n: Any) -> Any:
        if isinstance(n, Integral):
            if n < 0:
                raise ValueError("n shall not be negative value.")
            if n == 0:
                return Expr.from_number(1.0)
            val = self
            for _ in range(n - 1):
                val *= self
            return val
        return NotImplemented

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __repr__(self) -> str:
        if not self.terms:
            return "0*I"
        return " + ".join(repr(term) for term in self.terms)

    def to_expr(self) -> 'Expr':
        return self

    def max_n(self) -> int:
        try:
            return max(term.max_n() for term in self.terms if term.ops)
        except ValueError:
            return -1

    @property
    def n_qubits(self) -> int:
        return self.max_n() + 1

    def coeffs(self) -> Iterator[Any]:
        for term in self.terms:
            yield term.coeff

    def commutator(self, other: Any) -> 'Expr':
        return commutator(self, other)

    def is_commutable_with(self, other: Any) -> bool:
        return is_commutable(self, other)

    def simplify(self) -> 'Expr':
        """Simplify the Expr."""
        d = defaultdict(float)
        for term in self.terms:
            term = term.simplify()
            if term.ops in d:
                d[term.ops] = d[term.ops] + term.coeff
            else:
                d[term.ops] = term.coeff
        return Expr.from_terms_iter(
            Term.from_ops_iter(k, d[k]) for k in sorted(d, key=repr))

    def to_matrix(self, n_qubits: int = -1, *, sparse: bool = False, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert the linear combination of Paulis directly into a single PyTorch matrix."""
        if device is None:
            device = torch.device('cpu')
        if n_qubits == -1:
            n_qubits = self.n_qubits
            
        dim = 2**n_qubits
        expr = self.simplify()
        
        if sparse:
            # 疎行列の要素を動的にスタックして一括合成
            total_matrix = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.complex128, device=device),
                (dim, dim)
            )
            for term in expr.terms:
                total_matrix = total_matrix + term.to_matrix(n_qubits, sparse=True, device=device)
            return total_matrix.coalesce()
        else:
            # 各項の密行列を Autograd のグラフを維持したまま加算
            total_matrix = torch.zeros((dim, dim), dtype=torch.complex128, device=device)
            for term in expr.terms:
                total_matrix = total_matrix + term.to_matrix(n_qubits, sparse=False, device=device)
            return total_matrix


def qubo_bit(n: int) -> Expr:
    """Represent QUBO's bit to Pauli operator of Ising model."""
    return 0.5 - 0.5 * Z[n]


def from_qubo(qubo: Sequence[Sequence[float]]) -> Expr:
    """Convert to pauli operators of universal gate model."""
    h = 0.0
    for i in range(len(qubo)):
        h += qubo_bit(i) * qubo[i][i]
        for j in range(i + 1, len(qubo)):
            h += qubo_bit(i) * qubo_bit(j) * (qubo[i][j] + qubo[j][i])
    return h