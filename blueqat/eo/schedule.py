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
"""Pulse schedules: the hardware-facing time-resolved view of an exchange
circuit.

`to_schedule` turns a sequence of exchange pulses (or a Circuit of `exch`
gates) into a JSON-compatible dict with explicit start times, packing pulses
on disjoint spin pairs in parallel (ASAP scheduling; pulses on disjoint pairs
commute, so this never changes the unitary). The format is designed to be
handed to pulse-level control stacks (e.g. spinQICK-style backends) or
submitted through `blueqat.cloud`.

Schema::

    {
      "format": "blueqat-eo-schedule",
      "version": "1",
      "n_spins": 6,
      "amplitude": 1.0,          # exchange integral J during a pulse
      "pulses": [
        {"start": 0.0, "duration": 3.14159, "pair": [0, 1], "theta": 3.14159},
        ...
      ],
      "total_duration": 12.56637
    }

Durations are theta / amplitude (constant-amplitude pulses: the pulse area
theta = J * t is what fixes the gate).
"""

import math
from typing import Any, Dict, List, Sequence, Union

from ..circuit import Circuit
from ..gate import ExchangeGate
from .sequences import Pulse

SCHEDULE_FORMAT = "blueqat-eo-schedule"
SCHEDULE_VERSION = "1"


def _pulses_of(source: Union[Circuit, Sequence[Pulse]]) -> (List[Pulse], int):
    if isinstance(source, Circuit):
        pulses: List[Pulse] = []
        for op in source.ops:
            if not isinstance(op, ExchangeGate):
                raise ValueError(
                    f"Only exchange pulses can be scheduled; found '{op.lowername}'. "
                    "Transpile with backend='eo' first.")
            i, j = op.targets
            pulses.append(((int(i), int(j)), float(op.theta)))
        return pulses, source.n_qubits
    pulses = [((int(p[0][0]), int(p[0][1])), float(p[1])) for p in source]
    n_spins = max((max(pair) for pair, _ in pulses), default=-1) + 1
    return pulses, n_spins


def to_schedule(source: Union[Circuit, Sequence[Pulse]],
                amplitude: float = 1.0,
                n_spins: int = 0) -> Dict[str, Any]:
    """Build a time-resolved pulse schedule with ASAP parallel packing.

    Each pulse starts as soon as both of its spins are free; pulses touching
    disjoint pairs run simultaneously. Relative order of pulses sharing a
    spin is preserved, so the scheduled unitary equals the sequential one.

    The exchange unitary is exactly 2*pi-periodic in the pulse area, so theta
    is canonicalized into [0, 2*pi) -- a negative area (e.g. from a daggered
    circuit) becomes the equivalent positive-duration pulse, and pulses whose
    area is a multiple of 2*pi (no-ops) are dropped."""
    if amplitude <= 0:
        raise ValueError('amplitude must be positive.')
    pulses, inferred = _pulses_of(source)
    n_spins = max(n_spins, inferred)

    two_pi = 2.0 * math.pi
    free_at = [0.0] * n_spins
    entries = []
    total = 0.0
    for (i, j), theta in pulses:
        theta = theta % two_pi
        if theta < 1e-12 or two_pi - theta < 1e-12:
            continue
        duration = theta / amplitude
        start = max(free_at[i], free_at[j])
        end = start + duration
        free_at[i] = free_at[j] = end
        total = max(total, end)
        entries.append({
            "start": start,
            "duration": duration,
            "pair": [i, j],
            "theta": theta,
        })

    return {
        "format": SCHEDULE_FORMAT,
        "version": SCHEDULE_VERSION,
        "n_spins": n_spins,
        "amplitude": amplitude,
        "pulses": entries,
        "total_duration": total,
    }


def from_schedule(schedule: Dict[str, Any]) -> Circuit:
    """Rebuild an exchange-pulse Circuit from a schedule dict.

    Pulses are replayed in order of start time (ties broken by list order);
    since only disjoint pairs ever overlap, this reproduces the original
    unitary exactly."""
    if schedule.get("format") != SCHEDULE_FORMAT:
        raise ValueError("Not a blueqat-eo-schedule dict.")
    if schedule.get("version") not in (SCHEDULE_VERSION, ):
        raise ValueError(f"Unknown schedule version: {schedule.get('version')}")
    c = Circuit(int(schedule["n_spins"]))
    entries = sorted(enumerate(schedule["pulses"]),
                     key=lambda kv: (kv[1]["start"], kv[0]))
    for _, p in entries:
        i, j = p["pair"]
        c.exch(float(p["theta"]))[int(i), int(j)]
    return c


def schedule_stats(schedule: Dict[str, Any]) -> Dict[str, float]:
    """Summary numbers: pulse count, serial vs scheduled duration, speedup."""
    serial = sum(p["duration"] for p in schedule["pulses"])
    total = schedule["total_duration"]
    return {
        "n_pulses": len(schedule["pulses"]),
        "serial_duration": serial,
        "scheduled_duration": total,
        "parallel_speedup": (serial / total) if total > 0 else 1.0,
    }
