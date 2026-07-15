"""Exchange-only (EO) spin qubits: run a logical circuit using nothing but
Heisenberg exchange pulses, as on silicon quantum-dot hardware.

Each logical qubit is encoded in 3 physical spins; the only native operation
is the exchange pulse `exch(theta)[i, j]` (theta = pi is a full SWAP). Logical
single-qubit gates take 3 pulses, and the serial Fong-Wandzura CNOT takes 28
nearest-neighbor pulses on the 6-spin chain. The differentiable synthesizer at
the end compiles an arbitrary SU(2) into just 4 constant-amplitude pulses
using PyTorch autograd.
"""
import math

import torch

import blueqat.eo  # registers the 'eo' backend
from blueqat import Circuit
from blueqat.eo import encoding, synthesize_1q
from blueqat.eo.sequences import sequence_to_circuit


if __name__ == "__main__":
    print("Exchange-only spin qubits")
    print("=" * 50)

    # 1. Transpile a logical Bell circuit into exchange pulses.
    logical = Circuit(2).h[0].cx[0, 1]
    physical = logical.run(backend='eo')
    print(f"Logical: h[0].cx[0,1]  ->  {len(physical.ops)} exchange pulses "
          f"on {physical.n_qubits} spins")
    print(f"Pulse count by gate: H=3, FW-CNOT=28")

    # 2. Run the pulses on the encoded |00>_L state and check the result.
    init = encoding.encode_state([(1, 0), (1, 0)])
    final = physical.run(initial=init)
    basis = encoding.two_qubit_codeword_basis('+', '+')
    amps = basis.conj().T.to(final.dtype) @ final
    bell = torch.tensor([1, 0, 0, 1], dtype=torch.complex128) / math.sqrt(2)
    fid = (torch.vdot(bell, amps).abs() ** 2).item()
    print(f"Logical Bell-state fidelity: {fid:.12f}")
    print(f"Leakage out of the encoded subspace: "
          f"{encoding.leakage(final, 0):.2e} / {encoding.leakage(final, 1):.2e}")
    assert fid > 1 - 1e-9

    # 3. Differentiable pulse synthesis: an arbitrary rotation in 4 pulses
    #    (the fixed table needs 7 pulses for RX = H RZ H).
    target = torch.tensor([[math.cos(0.4), -1j * math.sin(0.4)],
                           [-1j * math.sin(0.4), math.cos(0.4)]],
                          dtype=torch.complex128)  # RX(0.8)
    seq = synthesize_1q(target, n_pulses=4, seed=42)
    print(f"\nSynthesized RX(0.8) in {len(seq)} pulses:")
    for (i, j), theta in seq:
        print(f"  exch({theta:.6f}) on spins ({i}, {j})")

    from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
    import numpy as np
    u = torch.tensor(np.array(circuit_to_unitary(sequence_to_circuit(seq, 3))),
                     dtype=torch.complex128)
    L = encoding.logical_action(u, '+')
    print(f"Synthesis fidelity: {encoding.logical_fidelity(L, target):.12f}")

    # 4. Hardware-facing pulse schedule: ASAP packing runs pulses on disjoint
    #    spin pairs simultaneously, and the dict is JSON-ready for a control
    #    stack or cloud submission.
    from blueqat.eo import schedule_stats, to_schedule
    sched = to_schedule(physical)
    stats = schedule_stats(sched)
    print(f"\nPulse schedule: {stats['n_pulses']} pulses, "
          f"serial {stats['serial_duration']:.2f} -> "
          f"scheduled {stats['scheduled_duration']:.2f} "
          f"(speedup x{stats['parallel_speedup']:.2f})")
    print("OK: exchange pulses alone reproduce the logical circuit.")
