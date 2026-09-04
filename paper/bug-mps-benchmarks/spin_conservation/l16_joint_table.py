#!/usr/bin/env python3
"""Sixteen-site data for the joint-conservation table of subsection G.

One quench per model: the XXZ chain and the Haldane-Shastry chain, both of which conserve the
total magnetization, started from a tilted Neel state. The tilt is what gives the table
content: from the plain Neel state every intermediate state stays in one U(1) charge sector
and ``<S^z>`` is exact for free, whereas the tilted state spreads over several sectors, so the
compression damages ``<S^z>`` and conserving it becomes a property a method has or lacks.

Four arms per model and cap, all through the shipped configuration surface of
:class:`AnalogSimParams`:

- ``none``   -- no correction; the uncorrected drift both invariants accumulate.
- ``pinH``   -- ``conserve_energy`` alone, the manuscript's energy correction.
- ``seq``    -- energy and magnetization corrected one after the other
  (``conserve_joint=False``); each scalar correction perturbs the previous one.
- ``joint``  -- the coupled solve (``conserve_joint=True``).
- ``joint3`` -- the coupled solve over three targets, ``{H, S^x, S^z}``. On an SU(2)-symmetric
  chain every total-spin component is conserved and each is a sum of on-site operators, so
  ``S^x`` is admissible on the same footing as ``S^z``; this arm exercises the solve at p=3.
- ``joint4`` -- the coupled solve over the complete admissible set of an SU(2)-symmetric
  chain, ``{H, S^x, S^y, S^z}``.

Recorded per arm: the drift of ``<H>``, ``<S^z>``, ``<S^2>`` measured with independently
assembled sparse operators, the final infidelity against a dense ``expm_multiply`` reference,
and the peak bond. Results append to ``l16_joint_table.json`` arm by arm, so a partial run is
usable.

Run: ``uv run python paper/bug-mps-benchmarks/spin_conservation/l16_joint_table.py``
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
from scipy.sparse.linalg import expm_multiply

from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    EvolutionMode,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx  # ruff:ignore[module-import-not-at-top-of-file]

from mqt.yaqs.analog import evolution as _evolution  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.analog.analog_tjm import capture_conservation_target  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.analog.evolution import apply_unitary_evolution  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.core.methods import bug as _bug  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.core.methods.conservation import apply_conservation as _stock_correction

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS

LENGTH = 16
DT = 0.01
TOTAL_TIME = 6.0
SVD_THRESHOLD = 1e-12
KRYLOV_TOL = 1e-12
CONSERVE_TOL = 1e-13
MODELS = ("xxz", "hs")
CAPS = (32, 16)
VARIANTS = ("none", "pinH", "pinS2", "seq", "joint", "joint3", "joint4", "joint5", "jointS2")
SWEEPS = ("none", "k2", "full")

# Number of solves the correction performed over the run, counted for every arm.
SOLVES = {"n": 0}


def _swept_correction(sites: int) -> Any:  # noqa: ANN401
    """Return a correction that acts at several centres instead of one.

    The stock correction acts at the centre the compression leaves. This applies it at
    ``sites`` centres nearest that one, moving the centre by QR between them and returning
    it afterwards so the integrator's gauge contract still holds. A scalar solve is exact,
    so for a single observable every centre after the first is below tolerance and skipped;
    only the coupled solve, which iterates, can use the extra centres.

    Args:
        sites: Number of centres to visit, or ``0`` for the whole chain.

    Returns:
        A drop-in replacement for :func:`apply_conservation`.
    """

    def correction(state: Any, mpo: Any, target: Any) -> bool:  # noqa: ANN401
        entry = state.orthogonality_center
        span = state.length if sites == 0 else sites
        order = range(entry - span + 1, entry + 1) if entry > state.length // 2 else range(span - 1, -1, -1)
        visited = [site for site in order if 0 <= site < state.length]
        displaced = False
        for site in visited:
            if state.orthogonality_center != site:
                state.set_canonical_form(site)
            if _stock_correction(state, mpo, target):
                displaced = True
                SOLVES["n"] += 1
        if state.orthogonality_center != entry:
            state.set_canonical_form(entry)
        return displaced

    return correction


def install_correction(sweep: str) -> None:
    """Patch the correction the integrators call, so its solves can be counted.

    Args:
        sweep: ``"none"`` for the stock single-centre correction, ``"k2"`` for two
            centres, ``"full"`` for the whole chain.
    """

    def counted(state: Any, mpo: Any, target: Any) -> bool:  # noqa: ANN401
        displaced = _stock_correction(state, mpo, target)
        if displaced:
            SOLVES["n"] += 1
        return displaced

    hook = counted if sweep == "none" else _swept_correction(2 if sweep == "k2" else 0)
    _bug.apply_conservation = hook
    _evolution.apply_conservation = hook


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--caps", default=",".join(map(str, CAPS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--initial", default="tilted_neel", choices=fx.INITIAL_STATES)
    parser.add_argument("--integrator", default="bug", choices=("bug", "tdvp"))
    parser.add_argument("--sweep", default="none", choices=SWEEPS)
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--output", type=Path, default=HERE / "l16_joint_table.json")
    return parser.parse_args()


def make_params(variant: str, cap: int, integrator: str) -> AnalogSimParams:
    """Build the simulation parameters of one arm.

    Args:
        variant: One of :data:`VARIANTS`.
        cap: Bond cap.
        integrator: ``"bug"`` or ``"tdvp"``, selecting the evolution mode.

    Returns:
        The configured parameters; the conservation settings are the only difference
        between arms.
    """
    conserve_energy = variant in {"pinH", "seq", "joint", "joint3", "joint4", "joint5", "jointS2"}
    observables: dict[str, Any] | None = None
    if variant == "pinS2":
        # The inadmissible quantity on its own, with no energy target, so the solve sees a
        # single inadmissible observable rather than a coupled system containing one.
        observables = {"S2": fx.s2_mpo(LENGTH)}
    elif variant in {"seq", "joint"}:
        observables = {"Sz": fx.sz_mpo(LENGTH)}
    elif variant == "joint3":
        observables = {"Sx": fx.sx_mpo(LENGTH), "Sz": fx.sz_mpo(LENGTH)}
    elif variant == "joint4":
        observables = {"Sx": fx.sx_mpo(LENGTH), "Sy": fx.sy_mpo(LENGTH), "Sz": fx.sz_mpo(LENGTH)}
    # The admissible set plus the inadmissible S^2, to measure what including it costs the
    # four quantities the flow does conserve.
    elif variant == "joint5":
        observables = {
            "Sx": fx.sx_mpo(LENGTH),
            "Sy": fx.sy_mpo(LENGTH),
            "Sz": fx.sz_mpo(LENGTH),
            "S2": fx.s2_mpo(LENGTH),
        }
    # The energy together with the inadmissible S^2 alone. At p=2 the solve is well enough
    # conditioned to meet both constraints, which separates the effect of including S^2 from
    # the effect of enlarging the system to p=5.
    elif variant == "jointS2":
        observables = {"S2": fx.s2_mpo(LENGTH)}
    return AnalogSimParams(
        elapsed_time=DT,
        dt=DT,
        # apply_unitary_evolution dispatches on this, and AnalogSimParams defaults to TDVP,
        # so leaving it unset silently runs every arm under the wrong integrator.
        evolution_mode=EvolutionMode.TDVP if integrator == "tdvp" else EvolutionMode.BUG,
        max_bond_dim=cap,
        trunc_mode="relative_discarded_weight",
        svd_threshold=SVD_THRESHOLD,
        krylov_tol=KRYLOV_TOL,
        conserve_energy=conserve_energy,
        conserve_observables=observables,
        conserve_joint=variant != "seq",
        conserve_tol=CONSERVE_TOL,
        get_state=True,
    )


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


def infidelity(reference: np.ndarray, approximate: np.ndarray) -> float:
    """Return ``1 - |<a|b>|^2 / (<a|a><b|b>)``.

    Args:
        reference: Reference state vector.
        approximate: Approximate state vector.

    Returns:
        The infidelity, clipped at zero.
    """
    denominator = float(np.vdot(reference, reference).real * np.vdot(approximate, approximate).real)
    fidelity = float(abs(complex(np.vdot(reference, approximate))) ** 2 / denominator)
    return max(0.0, 1.0 - fidelity)


def main() -> int:
    """Run every arm and append each to the output file.

    Returns:
        Zero on success.
    """
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    caps = [int(item) for item in args.caps.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    steps = round(args.total_time / DT)

    payload: dict[str, Any] = {
        "protocol": {
            "length": LENGTH,
            "initial_state": args.initial,
            "tilt_angle": fx.TILT_ANGLE,
            "start_bond": fx.INITIAL_CHI,
            "dt": DT,
            "total_time": args.total_time,
            "steps": steps,
            "trunc_mode": "relative_discarded_weight",
            "svd_threshold": SVD_THRESHOLD,
            "krylov_tol": KRYLOV_TOL,
            "conserve_tol": CONSERVE_TOL,
        },
        "arms": [],
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))

    for model in models:
        operators = {
            "H": fx.sparse_hamiltonian(LENGTH, model),
            "Sx": fx.sparse_sx(LENGTH),
            "Sy": fx.sparse_sy(LENGTH),
            "Sz": fx.sparse_sz(LENGTH),
            "S2": fx.sparse_s2(LENGTH),
        }
        initial = fx.initial_state(LENGTH, args.initial)
        print(f"[{model}] dense reference to T={args.total_time}", flush=True)
        reference = expm_multiply((-1j * args.total_time) * operators["H"], initial.to_vec().copy())
        initial_values = measure(initial, operators)

        for cap in caps:
            for variant in variants:
                done = {
                    (arm["model"], arm["cap"], arm["variant"], arm.get("integrator", "bug"), arm.get("sweep", "none"))
                    for arm in payload["arms"]
                }
                if (model, cap, variant, args.integrator, args.sweep) in done:
                    continue
                state = deepcopy(initial)
                hamiltonian = fx.hamiltonian_mpo(LENGTH, model)
                params = make_params(variant, cap, args.integrator)
                target = capture_conservation_target(state, hamiltonian, None, params)
                SOLVES["n"] = 0
                install_correction(args.sweep)
                started = time.perf_counter()
                # The library's own dispatch, so this measures the shipped code path: BUG
                # corrects inside the step after each of its two compressions, 2-TDVP after
                # the completed sweep at the site-0 centre it leaves.
                try:
                    for _ in range(steps):
                        apply_unitary_evolution(state, hamiltonian, params, conservation_target=target)
                finally:
                    _bug.apply_conservation = _stock_correction
                    _evolution.apply_conservation = _stock_correction
                final = measure(state, operators)
                arm = {
                    "model": model,
                    "cap": cap,
                    "variant": variant,
                    "integrator": args.integrator,
                    "sweep": args.sweep,
                    "initial_values": initial_values,
                    "final_values": final,
                    "drift": {name: final[name] - initial_values[name] for name in operators},
                    "infidelity": infidelity(reference, state.to_vec()),
                    "max_bond": int(max(state.bond_dimensions())),
                    "solves": SOLVES["n"],
                    "wall_seconds": time.perf_counter() - started,
                }
                payload["arms"].append(arm)
                args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(
                    f"  {model:4s} {args.integrator:4s} chi={cap:2d} {variant:6s} sweep={args.sweep:4s}"
                    f"  dH={arm['drift']['H']:+.3e}  dSx={arm['drift']['Sx']:+.3e}"
                    f"  dSz={arm['drift']['Sz']:+.3e}"
                    f"  dS2={arm['drift']['S2']:+.3e}  infid={arm['infidelity']:.3e}"
                    f"  solves={arm['solves']}  ({arm['wall_seconds']:.0f}s)",
                    flush=True,
                )
    print(f"\n{len(payload['arms'])} arms in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
