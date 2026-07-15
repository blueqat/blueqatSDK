# blueqat 2.1.0 — Exchange-Only spin qubits, correctness overhaul & cloud groundwork

This release adds first-class support for **exchange-only (EO) silicon spin
qubits**, closes two full audit rounds of correctness fixes (including
physically wrong results, not just crashes), reaches feature parity with
other SDKs on everyday circuit APIs, and lays the groundwork for API-key
cloud access. The default branch is now `main`. The test suite grew from
1,357 to **1,588 tests**.

## 🧲 Exchange-only spin qubits (`blueqat.eo`)

Each logical qubit is encoded in 3 physical spins (decoherence-free
subsystem); every gate is built purely from Heisenberg exchange pulses — the
only native operation of semiconductor spin-qubit hardware.

- **`exch(theta)` gate**: U(θ) = exp(−iθ/2(SWAP−I)); θ = π is an exact SWAP.
  Runs natively on both simulation modes, autograd-friendly.
- **Encoding & verification** (`eo.encoding`): codewords for both gauge
  (total-Sz) sectors, leakage detection into the S=3/2 quadruplet, logical
  action extraction.
- **Analytic pulse tables** (`eo.sequences`): RZ = 1 pulse, X/H = 3 pulses,
  and the serial **Fong–Wandzura CNOT (28 nearest-neighbor pulses)**,
  verified to act as an exact CNOT in all four gauge sectors *including the
  phase* (true gauge independence), after eoqrid (MIT) and Weinstein et al.,
  Nature 615, 817 (2023).
- **`'eo'` transpiler backend**: `Circuit(2).h[0].cx[0,1].run(backend='eo')`
  → an exchange-pulse circuit on 3n spins. Encoded Bell state verified at
  fidelity 1.0 with ~1e-32 leakage.
- **Differentiable pulse synthesis** (`eo.optimizer`, PyTorch autograd end to
  end): any logical SU(2) in **4 constant-amplitude pulses** (fidelity
  > 1−1e-9); `synthesize_2q` re-calibrates drifted 2-qubit sequences — 0.05 rad
  perturbations on all 28 FW pulses recover an exact CNOT in ~1 s.
- **Pulse schedules** (`eo.schedule`): JSON-ready, versioned time-resolved
  schedules with ASAP parallel packing, round-trippable, cloud-compatible.
- `quantize_sequence` for discrete-duration (clock-tick) hardware constraints.

## 🛠 Correctness fixes (two audit rounds)

Physically wrong results (silent):
- `get_time_evolution()` implemented exp(+itP) instead of exp(−itP).
- Y-basis rotations in `get_energy()`/`get_time_evolution()` used RX(−π/2),
  flipping the sign of every term with an odd number of Y operators.

Crashes and broken utilities:
- `margolus` macro (NameError), `draw` macro (dead `ibmq` backend),
  `flatten()`/JSON serialization on 3-qubit gates, `to_qasm()` on
  sx/sxdg/crx/cry/crz/rxx/ryy/rzz/zz, `Circuit.dagger()` measurement
  handling, `backend='composer'`, drawer with tensor-valued gate parameters,
  `get_measurement_sampler`'s 2^24-category multinomial cap.

Silent misbehavior:
- 19 fixed gates accepted bogus parameters (e.g. `x(0.5)[0]`, and the QASM
  parser's `zz` angle was silently dropped) — now `ValueError`.

## ✨ SDK-parity features

- `Circuit.depth()`, `Circuit.count_ops()` (slice-expanded, barrier-aware)
- `Circuit.probs(qubits=None)` — differentiable measurement probabilities
  with marginalization (PennyLane-style)
- `Circuit.expect(hamiltonian)` — differentiable expectation values
- `iswap`/`iswapdg` gates, `barrier` operation (identity in simulation, real
  `barrier` statement in QASM output)

## ☁️ Cloud access groundwork (`blueqat.cloud`)

- API-key resolution: `configure()` > `BLUEQAT_API_KEY` env >
  `~/.blueqat/config.json` (owner-only 0600 permissions, key masked in repr)
- `backend='cloud'` submits the versioned JSON circuit schema; transport is
  injectable until the public endpoint ships.

## 🧪 Testing & examples

- New suites: gate identities, QFT vs DFT matrix, teleportation, dagger
  round-trips, statevector/tensornet cross-checks, time evolution vs
  expm(−itH), gate unitarity/dagger/fallback consistency, analytic-gradient
  checks in both modes, EO physics (48 tests).
- Examples: `grover_search.py`, `qft.py`, `teleportation.py`,
  `exchange_only.py` added (all self-verifying).

## ⚠️ Notes

- The default branch was renamed `master` → `main`.
- `get_time_evolution()` now follows the standard exp(−itP) convention; QAOA
  results are unaffected (optimizers absorb the sign), but code relying on
  the old inverted convention should negate its time parameter.
- Fixed gates now reject spurious parameters; previously-silent misuse like
  `x(0.5)[0]` raises `ValueError`.
