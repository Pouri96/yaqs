#!/usr/bin/env python3
"""Experiment-A addendum: the energy correction on the uncompressed six-site problem.

The manuscript's six-site runner drives ``bug_sweep`` through its own
``alternating_endpoint_step``, which performs no compression and no
normalization. ``_postprocess_bug_state`` therefore never runs on that path, and
the post-compression energy correction is structurally absent from it. Checking
"the corrected run fires zero times" against that schedule would be vacuous.

This script asks the meaningful version of the same question, on the production
``bug()`` path configured to discard nothing (``svd_threshold = 0``, no cap,
``normalize=False``): does an uncompressed step leave the correction inert?
Corollary 5 says the uncompressed half-sweep preserves ``<H>`` exactly, so the
relative guard should never open and the trajectory should be bit-identical.

The fixture, the dense reference, and the error measure are imported from the
manuscript runner so that no detail is restated here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE.parent / "six_site_dense_reference_2026-08-17" / "run_benchmark.py"

_spec = importlib.util.spec_from_file_location("six_site_runner", RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
_runner = importlib.util.module_from_spec(_spec)
sys.modules["six_site_runner"] = _runner
_spec.loader.exec_module(_runner)

from mqt.yaqs.core.data_structures.simulation_parameters import (  # noqa: E402
    AnalogSimParams,
    EvolutionMode,
)
from mqt.yaqs.core.methods import bug as bug_module  # noqa: E402
from mqt.yaqs.core.methods.bug import bug  # noqa: E402
from mqt.yaqs.core.methods.conservation import energy_expectation  # noqa: E402

DT_GRID = _runner.DT_GRID
TOTAL_TIME = _runner.TOTAL_TIME
KRYLOV_TOL = _runner.KRYLOV_TOL


class FiringCounter:
    """Wrap the correction so each call and each firing is counted."""

    def __init__(self) -> None:
        """Initialise the counters and capture the production function."""
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


def uncompressed_params(dt: float, *, conserve: bool) -> AnalogSimParams:
    """Return BUG parameters that discard nothing.

    Args:
        dt: Full physical timestep.
        conserve: Whether to arm the energy correction.

    Returns:
        Configured :class:`AnalogSimParams`.
    """
    return AnalogSimParams(
        elapsed_time=TOTAL_TIME,
        dt=dt,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=None,
        svd_threshold=0.0,
        krylov_tol=KRYLOV_TOL,
        conserve_energy=conserve,
        conserve_tol=1e-12,
    )


def evolve(mpo: Any, initial: Any, dt: float, *, conserve: bool) -> tuple[Any, int, int]:
    """Evolve with the production ``bug`` step, counting correction firings.

    Args:
        mpo: Hamiltonian MPO.
        initial: Initial MPS; not modified.
        dt: Full physical timestep.
        conserve: Whether to arm the energy correction.

    Returns:
        The evolved state, the number of hook calls, and the number of firings.
    """
    state = deepcopy(initial)
    params = uncompressed_params(dt, conserve=conserve)
    target = energy_expectation(state, mpo) if conserve else None
    steps = int(round(TOTAL_TIME / dt))
    with FiringCounter() as counter:
        for _ in range(steps):
            bug(state, mpo, params, normalize=False, energy_target=target)
    return state, counter.calls, counter.firings


def main() -> None:
    """Run the addendum over the manuscript's timestep grid."""
    mpo = _runner.fixture_mpo()
    dense = _runner.independent_dense_hamiltonian()
    initial = _runner.initial_state()
    reference = _runner.dense_reference(dense, initial.to_vec().copy(), TOTAL_TIME)
    target = energy_expectation(initial, mpo)

    rows: list[dict[str, Any]] = []
    for dt in DT_GRID:
        plain, _, _ = evolve(mpo, initial, dt, conserve=False)
        corrected, calls, firings = evolve(mpo, initial, dt, conserve=True)
        identical = all(
            a.shape == b.shape and np.array_equal(a, b)
            for a, b in zip(plain.tensors, corrected.tensors, strict=True)
        )
        rows.append({
            "dt": dt,
            "steps": int(round(TOTAL_TIME / dt)),
            "error_uncorrected": _runner.phase_aligned_state_error(reference, plain.to_vec()),
            "error_corrected": _runner.phase_aligned_state_error(reference, corrected.to_vec()),
            "hook_calls": calls,
            "firings": firings,
            "bit_identical": bool(identical),
            "energy_drift_uncorrected": abs(energy_expectation(plain, mpo) - target),
            "max_bond_uncorrected": plain.get_max_bond(),
            "max_bond_corrected": corrected.get_max_bond(),
        })

    print(f"o_0 = {target:.15f}   relative guard = {1e-12 * max(1.0, abs(target)):.3e}\n")
    print("| h | steps | e_psi off | e_psi on | hook calls | firings | bit-identical | |<H>-o0| off |")
    print("|---:|---:|---:|---:|---:|---:|:--:|---:|")
    for row in rows:
        print(
            f"| {row['dt']:.5f} | {row['steps']} | {row['error_uncorrected']:.6e} | "
            f"{row['error_corrected']:.6e} | {row['hook_calls']} | {row['firings']} | "
            f"{'yes' if row['bit_identical'] else 'NO'} | {row['energy_drift_uncorrected']:.3e} |"
        )

    payload = {"target_energy": target, "rows": rows}
    output = HERE / "six_site_correction_check.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
