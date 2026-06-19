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
"""Utilities for convenient circuit and quantum state operations."""

import cmath
from collections import Counter
from dataclasses import dataclass
import math
import typing
from typing import Any, Dict, Iterator, Tuple, Union
import warnings

import numpy as np

if typing.TYPE_CHECKING:
    from . import Circuit


@dataclass
class QAOAResult:
    """Result data class for QAOA optimization."""
    params: np.ndarray
    circuit: 'Circuit'


def qaoa(hamiltonian: Any, step: int, init: typing.Optional['Circuit'] = None, mixer: typing.Optional[Any] = None) -> Union[QAOAResult, ValueError]:
    """Execute Quantum Approximate Optimization Algorithm (QAOA)."""
    import scipy.optimize as optimize
    from . import Circuit

    hamiltonian = hamiltonian.to_expr().simplify()
    N = hamiltonian.max_n()
    
    time_evolutions_cost = [
        term.get_time_evolution() for term in hamiltonian
    ] 
    
    time_evolutions_mixer = [
        term.get_time_evolution() for term in mixer
    ] if mixer else []
        
    def f(params: np.ndarray) -> float:
        betas = params[:step]
        gammas = params[step:]
        if init is None:
            c = Circuit(N).h[:]
        else:
            c = init.copy()
    
        for beta, gamma in zip(betas, gammas):
            for evo in time_evolutions_cost:
                evo(c, gamma)
            if mixer is None:
                c.rx(beta)[:]
            else:
                for evo in time_evolutions_mixer:
                    evo(c, beta)
    
        # Default run with quimb backend
        return float(c.run(backend="quimb", hamiltonian=hamiltonian))
    
    # Random initial guess for beta and gamma parameters
    initial_guess = np.random.rand(step * 2) * np.pi * 2
    
    result = optimize.minimize(
        f, 
        initial_guess, 
        method="Powell",
        options={
            "ftol": 5.0e-2,
            "xtol": 5.0e-2,
            "maxiter": 1000
        }
    )
    
    if result.success:
        fitted_params = result.x
        betas = fitted_params[:step]
        gammas = fitted_params[step:]

        if init is None:
            circ = Circuit(N).h[:]
        else:
            circ = init.copy()
    
        for beta, gamma in zip(betas, gammas):
            for evo in time_evolutions_cost:
                evo(circ, gamma)
            if mixer is None:
                circ.rx(beta)[:]
            else:
                for evo in time_evolutions_mixer:
                    evo(circ, beta)
                    
        return QAOAResult(params=fitted_params, circuit=circ)
    else:
        return ValueError(result.message)


def to_inttuple(
    bitstr: Union[str, Counter, Dict[str, int]]
) -> Union[Tuple[int, ...], Counter, Dict[Tuple[int, ...], int]]:
    """Convert from bit string like '01011' to int tuple like (0, 1, 0, 1, 1).

    Args:
        bitstr (str, Counter, dict): String which is written in "0" or "1".
            If all keys are bitstr, Counter or dict can also be converted.

    Returns:
        tuple of int, Counter, dict: Converted bits.

    Raises:
        ValueError: If bitstr type is unexpected or bitstr contains illegal character.
    """
    if isinstance(bitstr, str):
        return tuple(int(b) for b in bitstr)
    if isinstance(bitstr, Counter):
        return Counter({tuple(int(b) for b in k): v for k, v in bitstr.items()})
    if isinstance(bitstr, dict):
        return {tuple(int(b) for b in k): v for k, v in bitstr.items()}
    raise ValueError("bitstr type shall be `str`, `Counter` or `dict`")


def ignore_global_phase(statevec: np.ndarray) -> np.ndarray:
    """Multiply e^-iθ to `statevec` where θ is a phase of first non-zero element.

    Args:
        statevec np.ndarray: Statevector.

    Returns:
        np.ndarray: Unified statevector ignoring global phase.
    """
    for q in statevec:
        if abs(q) > 1e-7:
            ang = abs(q) / q
            statevec *= ang
            break
    return statevec


def check_unitarity(mat: np.ndarray) -> bool:
    """Check whether mat is a unitary matrix."""
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        return False
    return np.allclose(mat @ mat.T.conjugate(), np.eye(shape[0]))


def circuit_to_unitary(circ: 'Circuit', *runargs: Any, **runkwargs: Any) -> np.ndarray:
    """Convert circuit to unitary matrix.

    .. deprecated::
       This feature is moved to `blueqat.circuit_funcs.circuit_to_unitary`.
    """
    warnings.warn(
        "blueqat.util.circuit_to_unitary is moved to "
        "blueqat.circuit_funcs.circuit_to_unitary.circuit_to_unitary.",
        DeprecationWarning,
        stacklevel=2
    )
    from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary as f
    return f(circ, *runargs, **runkwargs)


def calc_u_params(mat: np.ndarray) -> Tuple[float, float, float, float]:
    """Calculate U-gate parameters from a 2x2 unitary matrix."""
    assert mat.shape == (2, 2)
    assert check_unitarity(mat)
    gamma = cmath.phase(mat[0, 0])
    mat = mat * cmath.exp(-1j * gamma)
    theta = math.atan2(abs(mat[1, 0]), mat[0, 0].real) * 2.0
    phi_plus_lambda = cmath.phase(mat[1, 1])
    phi = cmath.phase(mat[1, 0]) % (2.0 * math.pi)
    lam = (phi_plus_lambda - phi) % (2.0 * math.pi)
    return theta, phi, lam, gamma


def sqrt_2x2_matrix(mat: np.ndarray) -> np.ndarray:
    """Returns square root of a 2x2 matrix.

    Reference: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    assert mat.shape == (2, 2)
    s = np.sqrt(np.linalg.det(mat))
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    if abs(t) < 1e-8:  # Avoid division by zero
        s = -s
        t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return (mat + s * np.eye(2)) / t


def gen_graycode(n: int) -> Iterator[int]:
    """Generate an iterator which returns Gray code."""
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