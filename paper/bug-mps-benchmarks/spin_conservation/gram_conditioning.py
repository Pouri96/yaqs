#!/usr/bin/env python3
"""Does the corrected-BUG plateau come from near-parallel gradients?

The middle panel of the manuscript's figure shows the corrected BUG curve sitting on a
plateau until ``t ~ 4`` and then dropping by four orders. The manuscript attributes that to
the covariance Jacobian of the joint solve being close to singular at early times, which is
an inference about the solve, not something the figure shows. This script measures it.

At entry to every joint solve of the ``joint4`` run it records, at the undisplaced center:

- ``cond``, the 2-norm condition number of the normalised Gram matrix
  ``J_ab = 2 Re<g_a, g_b>_F / <C, C>_F``;
- ``max_cos``, the largest ``|<g_a, g_b>| / (||g_a|| ||g_b||)`` over ``a < b``, which is 1
  exactly when two gradients are parallel;
- ``residual_in``, ``max_a |<O_a> - o_a|`` before the solve, and ``residual_out`` after it.

If the manuscript's explanation is right, ``cond`` and ``max_cos`` are large before ``t ~ 4``
and fall afterwards, and ``residual_out`` sits at the plateau height over the same interval.
If they are flat throughout, the explanation is wrong and the plateau has another cause.

Run: ``uv run python paper/bug-mps-benchmarks/spin_conservation/gram_conditioning.py``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx
from mqt.yaqs.analog.analog_tjm import capture_conservation_target
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods import conservation as cons
from mqt.yaqs.core.methods.bug import bug as bug_evolve

LENGTH = 16
MODEL = "xxx"
INITIAL = "tilted_neel"
DT = 0.01
TOTAL_TIME = 6.0
CAP = 32
SVD_THRESHOLD = 1e-12
KRYLOV_TOL = 1e-12
CONSERVE_TOL = 1e-13


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--output", type=Path, default=HERE / "gram_conditioning.json")
    return parser.parse_args()


def main() -> int:
    """Run the instrumented sweep and write the record.

    Returns:
        Zero on success.
    """
    args = parse_args()
    steps = round(args.total_time / DT)

    records: list[dict[str, float]] = []
    clock = {"step": 0}
    original = cons.restore_invariants_joint

    def instrumented(state: Any, mpos: Any, targets: Any, **kwargs: Any) -> bool:
        """Record the Gram conditioning at the undisplaced center, then run the real solve."""
        center = state.orthogonality_center
        tensor = state.tensors[center]
        norm_squared = float(np.vdot(tensor, tensor).real)
        gradients, drift = [], []
        for mpo, goal in zip(mpos, targets, strict=False):
            projected = cons._center_operator(state, mpo, center)(tensor)  # noqa: SLF001
            value = float(np.vdot(tensor, projected).real) / norm_squared
            gradients.append(projected - value * tensor)
            drift.append(value - goal)
        gram = np.array(
            [[2.0 * float(np.vdot(a, b).real) / norm_squared for b in gradients] for a in gradients],
            dtype=np.float64,
        )
        norms = np.array([float(np.linalg.norm(g)) for g in gradients])
        cosines = [
            abs(float(np.vdot(gradients[i], gradients[j]).real)) / (norms[i] * norms[j])
            for i in range(len(gradients))
            for j in range(i + 1, len(gradients))
            if norms[i] > 0.0 and norms[j] > 0.0
        ]
        residual_in = float(np.max(np.abs(drift)))

        displaced = original(state, mpos, targets, **kwargs)

        # Re-measure after the solve, on the displaced center.
        tensor_out = state.tensors[center]
        norm_out = float(np.vdot(tensor_out, tensor_out).real)
        residual_out = 0.0
        for mpo, goal in zip(mpos, targets, strict=False):
            projected = cons._center_operator(state, mpo, center)(tensor_out)  # noqa: SLF001
            value = float(np.vdot(tensor_out, projected).real) / norm_out
            residual_out = max(residual_out, abs(value - goal))

        records.append({
            "time": clock["step"] * DT,
            "cond": float(np.linalg.cond(gram)),
            "max_cos": max(cosines) if cosines else 0.0,
            "min_grad_norm": float(norms.min()),
            "residual_in": residual_in,
            "residual_out": residual_out,
            "displaced": bool(displaced),
        })
        return displaced

    cons.restore_invariants_joint = instrumented
    try:
        state = fx.initial_state(LENGTH, INITIAL)
        hamiltonian = fx.hamiltonian_mpo(LENGTH, MODEL)
        params = AnalogSimParams(
            elapsed_time=DT,
            dt=DT,
            max_bond_dim=CAP,
            trunc_mode="relative_discarded_weight",
            svd_threshold=SVD_THRESHOLD,
            krylov_tol=KRYLOV_TOL,
            conserve_energy=True,
            conserve_observables={
                "Sx": fx.sx_mpo(LENGTH),
                "Sy": fx.sy_mpo(LENGTH),
                "Sz": fx.sz_mpo(LENGTH),
            },
            conserve_joint=True,
            conserve_tol=CONSERVE_TOL,
            get_state=True,
        )
        target = capture_conservation_target(deepcopy(state), hamiltonian, None, params)
        started = time.perf_counter()
        for step in range(1, steps + 1):
            clock["step"] = step
            bug_evolve(state, hamiltonian, params, conservation_target=target)
        elapsed = time.perf_counter() - started
    finally:
        cons.restore_invariants_joint = original

    payload = {
        "protocol": {
            "length": LENGTH,
            "model": MODEL,
            "initial_state": INITIAL,
            "dt": DT,
            "total_time": args.total_time,
            "cap": CAP,
            "targets": ["H", "Sx", "Sy", "Sz"],
            "note": "cond and max_cos are measured at the undisplaced center, before each solve.",
        },
        "wall_seconds": elapsed,
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)} solves recorded in {args.output} ({elapsed:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
