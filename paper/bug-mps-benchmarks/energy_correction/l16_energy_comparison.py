#!/usr/bin/env python3
"""Table-II analogue comparing BUG with and without the energy correction.

Same two L=16 models, initial states, MPOs, truncation settings and exact
references as the manuscript's matched-parameter benchmark; the fixtures are
imported from that runner rather than restated, so nothing can drift between the
two. The variants differ only in the correction:

    bug          conserve_energy off
    bug_ec_1e-12 conserve_energy on, the library default relative guard
    bug_ec_1e-14 conserve_energy on, a tighter guard

Two guards are run because ``conserve_tol`` is relative to ``|o_0|``: at
``|o_0| ~ 15`` the default 1e-12 gives a skip threshold of 1.5e-11, so the drift
column reports the guard rather than the accuracy of the correction unless the
guard is tightened. Running both shows which of the two is being measured.

Recorded per configuration: final infidelity (Eq. 34), phase-aligned error
(Eq. 33), the energy-drift trace ``|<H>(t) - o_0|``, the norm-drift trace, the
bond-dimension trace, the number of correction firings, and a single-shot wall
time. The wall time is not a median of warmed repetitions and must not be
compared against the manuscript's timing columns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE.parent / "l16_matched_optimized_2026-08-12" / "run_benchmark.py"

_spec = importlib.util.spec_from_file_location("l16_runner", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
_runner = importlib.util.module_from_spec(_spec)
sys.modules["l16_runner"] = _runner
_spec.loader.exec_module(_runner)

from mqt.yaqs.core.data_structures.simulation_parameters import (  # noqa: E402
    AnalogSimParams,
    EvolutionMode,
)
from mqt.yaqs.core.methods import bug as bug_module  # noqa: E402
from mqt.yaqs.core.methods.bug import bug  # noqa: E402
from mqt.yaqs.core.methods.conservation import energy_expectation  # noqa: E402
from mqt.yaqs.core.methods.tdvp import primitives  # noqa: E402

DEFAULT_TOLS = (1e-13,)



def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="tfim,hs", help="Comma-separated subset of tfim,hs")
    parser.add_argument("--dts", default=",".join(str(dt) for dt in _runner.DT_GRID))
    parser.add_argument("--total-time", type=float, default=_runner.TOTAL_TIME)
    parser.add_argument("--trace-every", type=int, default=10, help="Steps between trace samples")
    parser.add_argument(
        "--max-bond",
        type=int,
        default=_runner.MAX_BOND,
        help=(
            "Hard bond cap. The manuscript's Table II uses 512, which is inactive; "
            "its Table III sweeps 32, 64, 96, where the cap binds and truncation is "
            "the dominant error source."
        ),
    )
    parser.add_argument(
        "--tols",
        default=",".join(str(tol) for tol in DEFAULT_TOLS),
        help="Comma-separated conserve_tol values to run the corrected variant at.",
    )
    parser.add_argument("--output", type=Path, default=HERE / "l16_energy_comparison.json")
    return parser.parse_args()


def build_variants(tols: list[float]) -> list[tuple[str, bool, float]]:
    """Return the uncorrected variant followed by one corrected variant per tolerance.

    Returns:
        ``(label, conserve_energy, conserve_tol)`` triples.
    """
    variants: list[tuple[str, bool, float]] = [("bug", False, tols[0])]
    variants.extend((f"bug_ec_{tol:.0e}", True, tol) for tol in tols)
    return variants


def step_parameters(dt: float, *, conserve: bool, tol: float, max_bond: int) -> AnalogSimParams:
    """Return single-step BUG parameters matching the manuscript settings.

    Args:
        dt: Full physical timestep.
        conserve: Whether to arm the energy correction.
        tol: Relative guard of the correction.
        max_bond: Hard bond cap.

    Returns:
        Configured :class:`AnalogSimParams`.
    """
    return AnalogSimParams(
        elapsed_time=dt,
        dt=dt,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=max_bond,
        trunc_mode="relative_discarded_weight",
        svd_threshold=_runner.THRESHOLD,
        krylov_tol=_runner.KRYLOV_TOL,
        get_state=True,
        conserve_energy=conserve,
        conserve_tol=tol,
    )


class FiringCounter:
    """Count calls to the correction and how many of them displaced the centre."""

    def __init__(self) -> None:
        """Initialise counters and capture the production function."""
        self.calls = 0
        self.firings = 0
        self._original = bug_module.restore_energy_at_center

    def __enter__(self) -> FiringCounter:
        """Install the counting wrapper.

        Returns:
            This counter.
        """

        def wrapper(state: Any, mpo: Any, target: float, *, tol: float) -> bool:
            self.calls += 1
            fired = self._original(state, mpo, target, tol=tol)
            self.firings += int(fired)
            return fired

        bug_module.restore_energy_at_center = wrapper
        return self

    def __exit__(self, *_exc: object) -> None:
        """Restore the production function."""
        bug_module.restore_energy_at_center = self._original


def evolve(
    initial: Any,
    mpo: Any,
    *,
    dt: float,
    steps: int,
    conserve: bool,
    tol: float,
    trace_every: int,
    max_bond: int,
) -> dict[str, Any]:
    """Evolve one variant to the final time, recording traces.

    Args:
        initial: Padded initial MPS; not modified.
        mpo: Hamiltonian MPO.
        dt: Full physical timestep.
        steps: Number of physical steps.
        conserve: Whether to arm the energy correction.
        tol: Relative guard of the correction.
        trace_every: Sampling stride for the traces.
        max_bond: Hard bond cap.

    Returns:
        The final state vector, traces, firing counts, and wall time.
    """
    state = deepcopy(initial)
    local_mpo = deepcopy(mpo)
    params = step_parameters(dt, conserve=conserve, tol=tol, max_bond=max_bond)
    target = energy_expectation(state, local_mpo)

    times: list[float] = []
    energy_drift: list[float] = []
    norm_drift: list[float] = []
    max_bond_trace: list[int] = []

    # Diagnostics are excluded from the wall time: sampling the trace costs a
    # full environment contraction, which at the Haldane-Shastry ranks is
    # comparable to a step.
    wall = 0.0
    with FiringCounter() as counter:
        for step in range(steps):
            start = time.perf_counter()
            bug(state, local_mpo, params, energy_target=target if conserve else None)
            wall += time.perf_counter() - start
            if (step + 1) % trace_every == 0 or step == steps - 1:
                times.append((step + 1) * dt)
                energy_drift.append(abs(energy_expectation(state, local_mpo) - target))
                norm_drift.append(abs(float(state.scalar_product(state).real) - 1.0))
                max_bond_trace.append(int(state.get_max_bond()))

    return {
        "target_energy": target,
        "vector": state.to_vec(),
        "times": times,
        "energy_drift": energy_drift,
        "norm_drift": norm_drift,
        "max_bond": max_bond_trace,
        "final_bond_profile": _runner.bond_profile(state),
        "hook_calls": counter.calls,
        "firings": counter.firings,
        "wall_seconds": wall,
    }


def main() -> None:
    """Run the comparison over the requested grid."""
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    dts = [float(item) for item in args.dts.split(",") if item.strip()]
    variants = build_variants([float(item) for item in args.tols.split(",") if item.strip()])

    # One shared matrix-free adaptive Lanczos for every local exponential, as in
    # the manuscript runner.
    primitives.DENSE_THRESHOLD = -1

    payload: dict[str, Any] = {"total_time": args.total_time, "max_bond_dim": args.max_bond, "results": []}
    for model in models:
        print(f"[{model}] building fixtures and exact reference", flush=True)
        mpo = _runner.direct_ising_mpo() if model == "tfim" else _runner.direct_haldane_shastry_mpo()
        initial = _runner.padded_initial_state(model)
        hamiltonian = _runner.exact_sparse_hamiltonian(model)
        reference = np.asarray(
            expm_multiply(-1j * args.total_time * hamiltonian, initial.to_vec()),
            dtype=np.complex128,
        )

        for dt in dts:
            steps = int(round(args.total_time / dt))
            for name, conserve, tol in variants:
                result = evolve(
                    initial,
                    mpo,
                    dt=dt,
                    steps=steps,
                    conserve=conserve,
                    tol=tol,
                    trace_every=args.trace_every,
                    max_bond=args.max_bond,
                )
                vector = result.pop("vector")
                row = {
                    "model": model,
                    "dt": dt,
                    "steps": steps,
                    "variant": name,
                    "conserve_energy": conserve,
                    "conserve_tol": tol,
                    "max_bond_dim": args.max_bond,
                    "infidelity": _runner.infidelity(reference, vector),
                    "phase_error": _runner.phase_error(reference, vector),
                    **result,
                }
                payload["results"].append(row)
                print(
                    f"  {model} h={dt:<8g} {name:<13} I={row['infidelity']:.3e} "
                    f"drift={row['energy_drift'][-1]:.3e} chi={row['max_bond'][-1]:<4d} "
                    f"fired={row['firings']}/{row['hook_calls']} {row['wall_seconds']:.1f}s",
                    flush=True,
                )

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
