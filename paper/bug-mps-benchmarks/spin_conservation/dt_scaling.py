#!/usr/bin/env python3
"""Time-step scaling of the drift the uncompressed sweep accumulates.

This is the decisive test of the admissibility proposition. The drift of an observable over
a step splits into the part accumulated by the uncompressed half-sweeps and the part
accumulated by the compressions, and ``bug()`` already exposes both through its ``checkpoint``
callback. Only the first part is a property of the projected flow, and the proposition
predicts how each observable's share behaves as the step size is reduced at fixed horizon:

- ``H`` -- the flow conserves it exactly, so the sweep share sits at round-off at every step
  size, with no scaling at all.
- ``S^z`` -- a sum of on-site operators, so its action on the state is tangent and the flow
  conserves it exactly; whatever the sweep accumulates is time-discretization error and must
  vanish with the step size.
- ``S^2`` -- not a sum of on-site terms, so the projected flow itself moves it at a nonzero
  rate; the accumulated share must converge to a nonzero constant as the step size falls.

A single step size cannot tell the second case from the third, which is why this sweep exists.
The horizon is held fixed so the accumulated shares are comparable across step sizes.

Run: ``uv run python paper/bug-mps-benchmarks/spin_conservation/dt_scaling.py``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx  # ruff:ignore[module-import-not-at-top-of-file]

from mqt.yaqs.core.methods.bug import bug as bug_evolve  # ruff:ignore[module-import-not-at-top-of-file]

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS

LENGTH = 6
MODEL = "xxx"
INITIAL = "tilted_neel"
TOTAL_TIME = 2.0
SVD_THRESHOLD = 1e-12
KRYLOV_TOL = 1e-14
DTS = (0.04, 0.02, 0.01, 0.005, 0.0025)
CAPS = (2, 4)
OBSERVABLES = ("H", "Sz", "S2")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dts", default=",".join(map(str, DTS)))
    parser.add_argument("--caps", default=",".join(map(str, CAPS)))
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--output", type=Path, default=HERE / "dt_scaling.json")
    return parser.parse_args()


def measure(state: MPS, operators: dict[str, Any], *, reflected: bool) -> dict[str, float]:
    """Return the normalized sparse expectation values of ``state``.

    Args:
        state: The MPS to measure.
        operators: Sparse observables keyed by name.
        reflected: Whether ``state`` is currently in the reflected frame.

    Returns:
        The expectation values.
    """
    snapshot = deepcopy(state)
    if reflected:
        snapshot.flip_network()
    vector = snapshot.to_vec()
    norm_squared = float(np.vdot(vector, vector).real)
    return {name: float(np.vdot(vector, op @ vector).real) / norm_squared for name, op in operators.items()}


def run_cell(cap: int, dt: float, total_time: float, operators: dict[str, Any], initial: MPS) -> dict[str, Any]:
    """Accumulate the sweep and compression shares of the drift for one cell.

    Args:
        cap: Bond cap.
        dt: Physical step size.
        total_time: Horizon, held fixed across step sizes.
        operators: Sparse observables keyed by name.
        initial: The initial state; copied, not consumed.

    Returns:
        The recorded shares and the total drift.
    """
    state = deepcopy(initial)
    mpo = fx.hamiltonian_mpo(LENGTH, MODEL)
    params = AnalogSimParams(
        elapsed_time=dt,
        dt=dt,
        max_bond_dim=cap,
        trunc_mode="relative_discarded_weight",
        svd_threshold=SVD_THRESHOLD,
        krylov_tol=KRYLOV_TOL,
        get_state=True,
    )

    flow = dict.fromkeys(OBSERVABLES, 0.0)
    compression = dict.fromkeys(OBSERVABLES, 0.0)
    stage_values: dict[str, dict[str, float]] = {}

    def checkpoint(stage: str, checkpoint_state: MPS, *, reflected: bool) -> None:
        stage_values[stage] = measure(checkpoint_state, operators, reflected=reflected)

    initial_values = measure(state, operators, reflected=False)
    steps = round(total_time / dt)
    started = time.perf_counter()
    for _step in range(steps):
        previous = measure(state, operators, reflected=False)
        stage_values.clear()
        bug_evolve(state, mpo, params, checkpoint=checkpoint)
        # Sweep and compression alternate twice per step; each sweep is charged against the
        # state that entered it and each compression against the sweep before it.
        for sweep_stage, compression_stage in (
            ("first_half_sweep", "first_compression"),
            ("second_half_sweep", "second_compression"),
        ):
            for name in OBSERVABLES:
                flow[name] += stage_values[sweep_stage][name] - previous[name]
                compression[name] += stage_values[compression_stage][name] - stage_values[sweep_stage][name]
            previous = stage_values[compression_stage]

    final = measure(state, operators, reflected=False)
    return {
        "cap": cap,
        "dt": dt,
        "steps": steps,
        "flow_share": flow,
        "compression_share": compression,
        "drift": {name: final[name] - initial_values[name] for name in OBSERVABLES},
        "max_bond": int(max(state.bond_dimensions())),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> int:
    """Run every cell of the step-size sweep.

    Returns:
        Zero on success.
    """
    args = parse_args()
    dts = [float(item) for item in args.dts.split(",") if item.strip()]
    caps = [int(item) for item in args.caps.split(",") if item.strip()]

    operators = {
        "H": fx.sparse_hamiltonian(LENGTH, MODEL),
        "Sz": fx.sparse_sz(LENGTH),
        "S2": fx.sparse_s2(LENGTH),
    }
    initial = fx.initial_state(LENGTH, INITIAL)

    payload: dict[str, Any] = {
        "protocol": {
            "length": LENGTH,
            "model": MODEL,
            "initial_state": INITIAL,
            "tilt_angle": fx.TILT_ANGLE,
            "start_bond": fx.INITIAL_CHI,
            "total_time": args.total_time,
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "note": "flow_share is the drift accumulated by the uncompressed half-sweeps; "
            "compression_share is the drift accumulated by the compressions.",
        },
        "initial_values": measure(initial, operators, reflected=False),
        "cells": [],
    }

    for cap in caps:
        for dt in dts:
            cell = run_cell(cap, dt, args.total_time, operators, initial)
            payload["cells"].append(cell)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"  chi={cap:2d} dt={dt:<7g} steps={cell['steps']:4d}"
                f"  flow: H={cell['flow_share']['H']:+.3e}"
                f" Sz={cell['flow_share']['Sz']:+.3e}"
                f" S2={cell['flow_share']['S2']:+.3e}"
                f"  ({cell['wall_seconds']:.0f}s)",
                flush=True,
            )

    print(f"\n{len(payload['cells'])} cells in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
