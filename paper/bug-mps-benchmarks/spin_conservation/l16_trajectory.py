#!/usr/bin/env python3
"""Time-resolved invariant drift for the figure of the standalone conservation paper.

Same protocol as ``l16_joint_table.py`` -- isotropic Heisenberg chain at ``L=16`` from the
tilted Neel state -- but the invariants are measured every ``--record-every`` physical steps
instead of only at the horizon, so the drift can be plotted against time.

The default arms are the seven rows of the excluded-invariant table under the BUG
composition: the uncorrected run, and each of the restored sets ``{H, S^a}``,
``{H, S^2}`` and ``{H, S^a, S^2}`` with the correction applied at the one centre the
compression leaves and again at every centre in turn.

``<S^2>`` is recorded in every arm whether or not it is restored, since its drift comes from
the projected flow rather than from the compression.

Run: ``uv run python paper/bug-mps-benchmarks/spin_conservation/l16_trajectory.py``
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx  # ruff:ignore[module-import-not-at-top-of-file]
import l16_joint_table as jt  # ruff:ignore[module-import-not-at-top-of-file]

from mqt.yaqs.analog.analog_tjm import capture_conservation_target  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.analog.evolution import apply_unitary_evolution  # ruff:ignore[module-import-not-at-top-of-file]

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS

# The arm definitions live in l16_joint_table so the trajectories and the tables cannot
# drift apart. A second copy here once omitted "joint5" and would have recorded that arm
# with no correction installed at all.
LENGTH = jt.LENGTH
DT = jt.DT
TOTAL_TIME = jt.TOTAL_TIME
CAP = 32
MODEL = "xxx"
SVD_THRESHOLD = jt.SVD_THRESHOLD
KRYLOV_TOL = jt.KRYLOV_TOL
CONSERVE_TOL = jt.CONSERVE_TOL
RECORD_EVERY = 10
ARMS = (
    ("bug", "none", "none"),
    ("bug", "joint4", "none"),
    ("bug", "joint4", "full"),
    ("bug", "jointS2", "none"),
    ("bug", "jointS2", "full"),
    ("bug", "joint5", "none"),
    ("bug", "joint5", "full"),
)


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--record-every", type=int, default=RECORD_EVERY)
    parser.add_argument("--output", type=Path, default=HERE / "l16_trajectory.json")
    parser.add_argument(
        "--arms",
        default=",".join(f"{i}:{v}:{s}" for i, v, s in ARMS),
        help='Comma-separated "integrator:variant:sweep" triples, e.g. "bug:joint5:full". '
        "The sweep field may be omitted and then defaults to the single-centre correction.",
    )
    return parser.parse_args()


def measure(state: MPS, operators: dict[str, Any]) -> dict[str, float]:
    """Return the normalized sparse expectation values of ``state``.

    Args:
        state: The MPS to measure.
        operators: Sparse observables keyed by name.

    Returns:
        The expectation values.
    """
    vector = state.to_vec()
    norm_squared = float(np.vdot(vector, vector).real)
    return {name: float(np.vdot(vector, op @ vector).real) / norm_squared for name, op in operators.items()}


def main() -> int:
    """Run every arm and append its trajectory to the output file.

    Returns:
        Zero on success.
    """
    args = parse_args()
    steps = round(args.total_time / DT)

    operators = {
        "H": fx.sparse_hamiltonian(LENGTH, MODEL),
        "Sx": fx.sparse_sx(LENGTH),
        "Sy": fx.sparse_sy(LENGTH),
        "Sz": fx.sparse_sz(LENGTH),
        "S2": fx.sparse_s2(LENGTH),
    }
    initial = fx.initial_state(LENGTH, "tilted_neel")
    initial_values = measure(initial, operators)

    payload: dict[str, Any] = {
        "protocol": {
            "length": LENGTH,
            "model": MODEL,
            "initial_state": "tilted_neel",
            "tilt_angle": fx.TILT_ANGLE,
            "start_bond": fx.INITIAL_CHI,
            "dt": DT,
            "total_time": args.total_time,
            "steps": steps,
            "cap": args.cap,
            "record_every": args.record_every,
            "trunc_mode": "relative_discarded_weight",
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "conserve_tol": CONSERVE_TOL,
        },
        "initial_values": initial_values,
        "arms": [],
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))

    arms = []
    for item in args.arms.split(","):
        if not item.strip():
            continue
        fields = item.split(":")
        arms.append((fields[0], fields[1], fields[2] if len(fields) > 2 else "none"))

    for integrator, variant, sweep in arms:
        # Arms written before the sweep field existed are single-centre runs.
        done = {(a["integrator"], a["variant"], a.get("sweep", "none")) for a in payload["arms"]}
        if (integrator, variant, sweep) in done:
            print(f"  skip {integrator}/{variant}/{sweep} (already recorded)", flush=True)
            continue

        state = deepcopy(initial)
        hamiltonian = fx.hamiltonian_mpo(LENGTH, MODEL)
        params = jt.make_params(variant, args.cap, integrator)
        target = capture_conservation_target(state, hamiltonian, None, params)

        times = [0.0]
        series: dict[str, list[float]] = {name: [initial_values[name]] for name in operators}
        bonds = [int(max(state.bond_dimensions()))]

        jt.SOLVES["n"] = 0
        jt.install_correction(sweep)
        started = time.perf_counter()
        try:
            for step in range(1, steps + 1):
                # The library's own dispatch, so this measures the shipped code path: BUG
                # corrects inside the step after each of its two compressions, 2-TDVP after
                # the completed sweep at the site-0 centre it leaves.
                apply_unitary_evolution(state, hamiltonian, params, conservation_target=target)

                if step % args.record_every == 0 or step == steps:
                    values = measure(state, operators)
                    times.append(step * DT)
                    for name in operators:
                        series[name].append(values[name])
                    bonds.append(int(max(state.bond_dimensions())))
        finally:
            jt.install_correction("none")

        arm = {
            "integrator": integrator,
            "variant": variant,
            "sweep": sweep,
            "solves": jt.SOLVES["n"],
            "cap": args.cap,
            "times": times,
            "values": series,
            "drift": {name: [v - initial_values[name] for v in series[name]] for name in operators},
            "max_bond": bonds,
            "wall_seconds": time.perf_counter() - started,
        }
        payload["arms"].append(arm)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"  {integrator:4s} {variant:7s} sweep={sweep:4s} chi={args.cap:2d}"
            f"  solves={arm['solves']:5d}"
            f"  dH={arm['drift']['H'][-1]:+.3e}  dSx={arm['drift']['Sx'][-1]:+.3e}"
            f"  dSy={arm['drift']['Sy'][-1]:+.3e}  dSz={arm['drift']['Sz'][-1]:+.3e}"
            f"  dS2={arm['drift']['S2'][-1]:+.3e}"
            f"  ({arm['wall_seconds']:.0f}s)",
            flush=True,
        )

    print(f"\n{len(payload['arms'])} arms in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
