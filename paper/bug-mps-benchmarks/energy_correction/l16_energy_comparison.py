#!/usr/bin/env python3
"""Table-II analogue comparing BUG, BUG with the energy correction, and 2-TDVP.

Same two L=16 models, initial states, MPOs, truncation settings and exact
references as the manuscript's matched-parameter benchmark; the fixtures are
imported from that runner rather than restated, so nothing can drift between the
two.

    bug          BUG, conserve_energy off
    tdvp2        two-site TDVP, the manuscript's baseline
    bug_ec_<tol> BUG with the correction armed at that relative guard

2-TDVP is an uncorrected baseline. It also compresses and therefore also loses
``<H>``, which is what makes the comparison worth drawing; but it truncates at
every local two-site split rather than in one global compression sweep, so it
offers no single orthogonality centre left by a compression for the correction to
act at. Applying the correction to it is not a configuration change.

``conserve_tol`` is relative to ``|o_0|``, so where the truncation is mild enough
for the guard to bind it is the guard, not the correction, that sets the restored
value. Passing several tolerances separates the two regimes.

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
from mqt.yaqs.core.methods.tdvp import tdvp as tdvp_evolve  # noqa: E402

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
    parser.add_argument("--no-tdvp", action="store_true", help="Skip the 2-TDVP baseline.")
    parser.add_argument("--output", type=Path, default=HERE / "l16_energy_comparison.json")
    return parser.parse_args()


def build_variants(tols: list[float], *, with_tdvp: bool) -> list[tuple[str, str, bool, float]]:
    """Return the baselines followed by one corrected BUG variant per tolerance.

    2-TDVP enters as an uncorrected baseline only. It truncates at every local
    two-site split rather than in one global compression sweep, so there is no
    single orthogonality centre left by a compression for the correction to act
    at; the no-error-reduction identity is stated for exactly that centre.

    Args:
        tols: Relative guards for the corrected variants.
        with_tdvp: Whether to include the 2-TDVP baseline.

    Returns:
        ``(label, method, conserve_energy, conserve_tol)`` tuples.
    """
    variants: list[tuple[str, str, bool, float]] = [("bug", "bug", False, tols[0])]
    if with_tdvp:
        variants.append(("tdvp2", "tdvp", False, tols[0]))
    variants.extend((f"bug_ec_{tol:.0e}", "bug", True, tol) for tol in tols)
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
        tdvp_mode="2site",
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
    method: str,
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
        method: ``bug`` or ``tdvp``.
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
            if method == "tdvp":
                tdvp_evolve(state, local_mpo, params)
            else:
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
    variants = build_variants(
        [float(item) for item in args.tols.split(",") if item.strip()],
        with_tdvp=not args.no_tdvp,
    )

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
            for name, method, conserve, tol in variants:
                result = evolve(
                    initial,
                    mpo,
                    dt=dt,
                    steps=steps,
                    method=method,
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
                    "method": method,
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
