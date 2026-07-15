Exchange-only spin qubits
=========================

:mod:`blueqat.eo` supports **exchange-only (EO) qubits**, the operating mode
of semiconductor (silicon quantum-dot) spin hardware: each logical qubit is
encoded in **3 physical spins**, and the *only* native operation is the
Heisenberg exchange pulse between two spins.

The exchange pulse
------------------

.. math::

   U(\theta) = e^{-i \frac{\theta}{2}(\mathrm{SWAP} - I)}

acts as identity on the triplet subspace and phases the singlet by
:math:`e^{i\theta}`; :math:`\theta = \pi` is an exact SWAP,
:math:`\theta = \pi/2` a square-root SWAP (up to phase). It is available on
every circuit as ``exch(theta)[i, j]`` and runs on both simulation modes with
full autograd support.

Encoding
--------

The logical codewords live in the total-spin :math:`S = 1/2` sector
(:math:`|0_L\rangle` uses the singlet of spins 0, 1):

.. code-block:: python

   from blueqat.eo import encoding

   state = encoding.encode_state([(1, 0), (0, 1)])   # |0>_L |1>_L on 6 spins
   encoding.leakage(state, triple=0)                 # population outside the code
   encoding.logical_action(u8x8)                     # 2x2 logical block of a 3-spin unitary

Each codeword exists in two *gauge* copies (total-Sz :math:`\pm 1/2`);
all shipped sequences act identically -- including the phase -- in every
gauge sector.

Transpiling logical circuits to pulses
--------------------------------------

Importing :mod:`blueqat.eo` registers the ``'eo'`` backend:

.. code-block:: python

   import blueqat.eo
   from blueqat import Circuit

   physical = Circuit(2).h[0].cx[0, 1].run(backend='eo')
   # 31 exch pulses on 6 spins: H = 3 pulses, Fong-Wandzura CNOT = 28

The Fong-Wandzura CNOT is the serial 28-pulse nearest-neighbor sequence
(Weinstein et al., *Nature* **615**, 817 (2023)); logical RZ costs 1 pulse
and X/H cost 3. The resulting circuit contains only ``exch`` gates and runs
on any simulation backend:

.. code-block:: python

   init = encoding.encode_state([(1, 0), (1, 0)])
   final = physical.run(initial=init)      # encoded Bell state, fidelity 1.0

Differentiable pulse synthesis
------------------------------

Because the simulator is torch-native, pulse sequences can be *optimized* by
gradient descent:

.. code-block:: python

   from blueqat.eo import synthesize_1q, synthesize_2q, quantize_sequence

   # Any SU(2) as 4 constant-amplitude pulses (fidelity > 1 - 1e-9)
   seq = synthesize_1q(target_2x2, n_pulses=4)

   # Re-calibrate a drifted 2-qubit sequence back to an exact gate,
   # gauge-independence enforced across all four total-Sz sectors
   refined = synthesize_2q(cx_4x4, pairs=pulse_pairs, initial_thetas=drifted)

   # Snap pulse areas to hardware clock ticks
   seq_q = quantize_sequence(seq, step=2 * 3.141592653589793 / 4096)

Pulse schedules
---------------

:func:`~blueqat.eo.to_schedule` converts pulses into a JSON-compatible,
time-resolved schedule with ASAP parallel packing (pulses on disjoint spin
pairs overlap; shared-spin order is preserved, so the unitary is unchanged):

.. code-block:: python

   from blueqat.eo import to_schedule, from_schedule, schedule_stats

   sched = to_schedule(physical)
   schedule_stats(sched)
   # {'n_pulses': 31, 'serial_duration': 94.2, 'scheduled_duration': 58.7,
   #  'parallel_speedup': 1.6}
   from_schedule(sched)      # back to a Circuit, unitary preserved

The schedule format is designed for pulse-level control stacks and for
submission through :doc:`the cloud backend <cloud>`.

Topology note
-------------

Emitted pulses assume any pair inside the triples involved in a gate can be
pulsed. Mapping onto strict nearest-neighbor-only hardware additionally
requires dot-orientation assignment and spin-level SWAP routing, which is
future work.
