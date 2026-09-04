#!/usr/bin/env python3
"""Do the unpinned observables improve when the pinned ones are restored?

The joint solve holds ``{H, S^a, S^2}`` at round-off and costs infidelity. Infidelity is a
single global number, so it does not say what happens to the quantities a practitioner
actually reports. This script measures, against the same dense ``expm_multiply`` reference,
four families that the solve never touches:

- the local magnetization profile ``<S^a_i>``, one value per site,
- the bond energy density ``<S_i . S_{i+1}>``, one value per bond, whose sum is the pinned
  ``<H>``,
- the longitudinal correlator ``<S^z_c S^z_{c+j}>`` from the chain centre,
- the staggered magnetization, a global quantity outside the admissible set.

Arms are built by importing :mod:`l16_joint_table`, so the evolution is bit-identical to the
runs behind the joint-conservation table and the two sets of numbers may be compared directly.

Run: ``uv run python paper/bug-mps-benchmarks/spin_conservation/l16_observables.py \
        --variants none,joint5 --sweep full``
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
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx  # ruff:ignore[module-import-not-at-top-of-file]
import l16_joint_table as jt  # ruff:ignore[module-import-not-at-top-of-file]

from mqt.yaqs.analog.analog_tjm import capture_conservation_target  # ruff:ignore[module-import-not-at-top-of-file]
from mqt.yaqs.analog.evolution import apply_unitary_evolution  # ruff:ignore[module-import-not-at-top-of-file]

LENGTH = jt.LENGTH
DT = jt.DT

_I2 = sp.identity(2, format="csr", dtype=complex)
_SX = sp.csr_matrix(np.array([[0.0, 0.5], [0.5, 0.0]], dtype=complex))
_SY = sp.csr_matrix(np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=complex))
_SZ = sp.csr_matrix(np.array([[0.5, 0.0], [0.0, -0.5]], dtype=complex))
_PAULI = {"x": _SX, "y": _SY, "z": _SZ}


def site_op(length: int, site: int, op: sp.spmatrix) -> sp.spmatrix:
    """Return ``op`` acting on one site of a chain, identity elsewhere.

    Args:
        length: Number of sites.
        site: Index of the site the operator acts on.
        op: The single-site operator.

    Returns:
        The sparse operator on the full space.
    """
    out = op if site == 0 else _I2
    for index in range(1, length):
        out = sp.kron(out, op if index == site else _I2, format="csr")
    return out


def build_observables(length: int) -> dict[str, list[tuple[str, sp.spmatrix]]]:
    """Return the unpinned observable families, keyed by family name.

    Args:
        length: Number of sites.

    Returns:
        Family name to a list of ``(label, operator)`` pairs.
    """
    families: dict[str, list[tuple[str, sp.spmatrix]]] = {}

    for axis, op in _PAULI.items():
        families[f"local_S{axis}"] = [(f"S{axis}_{i}", site_op(length, i, op)) for i in range(length)]

    bonds = []
    for i in range(length - 1):
        term = sum(
            site_op(length, i, op) @ site_op(length, i + 1, op) for op in _PAULI.values()
        )
        bonds.append((f"h_{i}", term.tocsr()))
    families["bond_energy"] = bonds

    centre = length // 2
    families["corr_zz"] = [
        (f"czz_{j}", (site_op(length, centre, _SZ) @ site_op(length, centre + j, _SZ)).tocsr())
        for j in range(1, length - centre)
    ]

    stag = sum((-1.0) ** i * site_op(length, i, _SZ) for i in range(length))
    families["staggered"] = [("Ms", (stag / length).tocsr())]
    return families


def check_conventions(length: int, families: dict[str, list[tuple[str, sp.spmatrix]]]) -> None:
    """Verify the site ordering against the already-trusted fixtures.

    The single-site operators built here must use the same index convention as
    ``state.to_vec()``. Summing them has to reproduce the fixture operators exactly, otherwise
    every number this script prints is measured in the wrong basis.

    Args:
        length: Number of sites.
        families: Output of :func:`build_observables`.

    Raises:
        AssertionError: If a convention check fails.
    """
    def sparse_max_abs(diff: sp.spmatrix) -> float:
        """Largest absolute entry, without ever densifying the operator."""
        csr = diff.tocsr()
        csr.eliminate_zeros()
        return float(abs(csr.data).max()) if csr.nnz else 0.0

    total_sz = sum(op for _, op in families["local_Sz"])
    delta_sz = sparse_max_abs(total_sz - fx.sparse_sz(length))
    total_h = sum(op for _, op in families["bond_energy"])
    delta_h = sparse_max_abs(total_h - fx.sparse_hamiltonian(length, "xxx"))
    print(f"[check] |sum_i S^z_i - fixture| = {delta_sz:.3e}", flush=True)
    print(f"[check] |sum_i h_i   - fixture| = {delta_h:.3e}", flush=True)
    assert delta_sz < 1e-12, f"site ordering disagrees with fixtures_n.sparse_sz ({delta_sz:.3e})"
    assert delta_h < 1e-12, f"bond terms disagree with fixtures_n.sparse_hamiltonian ({delta_h:.3e})"


def expectations(vector: np.ndarray, ops: list[tuple[str, sp.spmatrix]]) -> np.ndarray:
    """Return normalized expectation values of ``ops`` in ``vector``.

    Args:
        vector: State vector.
        ops: ``(label, operator)`` pairs.

    Returns:
        The expectation values in the order given.
    """
    norm_squared = float(np.vdot(vector, vector).real)
    return np.array([float(np.vdot(vector, op @ vector).real) / norm_squared for _, op in ops])


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="none,joint4,joint5")
    parser.add_argument("--sweep", default="none", choices=jt.SWEEPS)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--integrator", default="bug", choices=("bug", "tdvp"))
    parser.add_argument("--total-time", type=float, default=jt.TOTAL_TIME)
    parser.add_argument("--output", type=Path, default=HERE / "l16_observables.json")
    return parser.parse_args()


def main() -> int:
    """Run each arm and record the unpinned observable errors.

    Returns:
        Zero on success.
    """
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    steps = round(args.total_time / DT)

    families = build_observables(LENGTH)
    check_conventions(LENGTH, families)

    pinned = {
        "H": fx.sparse_hamiltonian(LENGTH, "xxx"),
        "Sx": fx.sparse_sx(LENGTH),
        "Sy": fx.sparse_sy(LENGTH),
        "Sz": fx.sparse_sz(LENGTH),
        "S2": fx.sparse_s2(LENGTH),
    }

    payload: dict[str, Any] = {
        "protocol": {
            "length": LENGTH,
            "cap": args.cap,
            "dt": DT,
            "total_time": args.total_time,
            "steps": steps,
            "integrator": args.integrator,
        },
        "arms": [],
    }
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))

    initial = fx.initial_state(LENGTH, "tilted_neel")
    print(f"[xxx] dense reference to T={args.total_time}", flush=True)
    reference = expm_multiply((-1j * args.total_time) * pinned["H"], initial.to_vec().copy())
    exact = {name: expectations(reference, ops) for name, ops in families.items()}

    for variant in variants:
        done = {(arm["variant"], arm["sweep"]) for arm in payload["arms"]}
        if (variant, args.sweep) in done:
            continue
        state = deepcopy(initial)
        hamiltonian = fx.hamiltonian_mpo(LENGTH, "xxx")
        params = jt.make_params(variant, args.cap, args.integrator)
        target = capture_conservation_target(state, hamiltonian, None, params)
        jt.SOLVES["n"] = 0
        jt.install_correction(args.sweep)
        started = time.perf_counter()
        try:
            for _ in range(steps):
                apply_unitary_evolution(state, hamiltonian, params, conservation_target=target)
        finally:
            jt.install_correction("none")
        wall = time.perf_counter() - started

        vector = state.to_vec()
        norm_squared = float(np.vdot(vector, vector).real)
        ref_norm_squared = float(np.vdot(reference, reference).real)

        record: dict[str, Any] = {
            "variant": variant,
            "sweep": args.sweep,
            "cap": args.cap,
            "integrator": args.integrator,
            "solves": jt.SOLVES["n"],
            "wall_seconds": wall,
            "infidelity": jt.infidelity(reference, vector),
            "pinned_error": {
                name: abs(
                    float(np.vdot(vector, op @ vector).real) / norm_squared
                    - float(np.vdot(reference, op @ reference).real) / ref_norm_squared
                )
                for name, op in pinned.items()
            },
            "families": {},
            "profiles": {},
        }
        for name, ops in families.items():
            got = expectations(vector, ops)
            err = np.abs(got - exact[name])
            record["families"][name] = {
                "max_abs_error": float(err.max()),
                "rms_abs_error": float(np.sqrt(np.mean(err**2))),
                "mean_abs_error": float(err.mean()),
            }
            record["profiles"][name] = {"exact": exact[name].tolist(), "arm": got.tolist()}

        payload["arms"].append(record)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = "  ".join(
            f"{name}={record['families'][name]['max_abs_error']:.3e}" for name in families
        )
        print(
            f"  {variant:8s} sweep={args.sweep:5s} I={record['infidelity']:.4e} "
            f"solves={record['solves']:5d}  {summary}  ({wall:.0f}s)",
            flush=True,
        )

    print(f"\n{len(payload['arms'])} arms in {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
