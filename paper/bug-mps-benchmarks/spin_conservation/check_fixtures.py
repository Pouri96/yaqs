#!/usr/bin/env python3
"""Validate the length-parametrized fixtures before any physics is run.

Every conclusion of the screen is downstream of these operators, and a subtly wrong ``S^2``
would fabricate the whole result rather than fail loudly. This script therefore checks, in
order:

1. Each MPO against an independently assembled sparse matrix, at ``L = 4`` and ``L = 6``.
2. The ``L = 16`` rebuilds against the manuscript's own operators, which the screen must not
   contradict.
3. The symmetry structure the screen relies on: ``[S^2, H] = 0`` exactly for the two
   SU(2)-symmetric models and not for the other two, and ``[S^z, H] = 0`` for all three spin
   chains.
4. Known spectra and expectation values that pin the normalization of ``S^2``.
5. The initial states: normalized, right-canonical with the centre at site 0, and reproducing
   the manuscript's state where they overlap.

Run with ``uv run python paper/bug-mps-benchmarks/spin_conservation/check_fixtures.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fixtures_n as fx  # ruff:ignore[module-import-not-at-top-of-file]

#: Absolute tolerance for operator comparisons. The constructions differ in contraction order
#: only, so agreement is at round-off; 1e-12 leaves headroom without hiding a real defect.
ATOL = 1e-12

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:  # ruff:ignore[boolean-type-hint-positional-argument]
    """Record one assertion without aborting the remaining checks.

    Args:
        condition: Whether the assertion holds.
        message: Description printed for the result.
    """
    status = "ok  " if condition else "FAIL"
    if not condition:
        _FAILURES.append(message)
    print(f"  [{status}] {message}", flush=True)


def max_abs(matrix: sparse.spmatrix | np.ndarray) -> float:
    """Return the largest absolute entry of a sparse or dense matrix.

    Args:
        matrix: The matrix to reduce.

    Returns:
        The maximum absolute entry, or ``0.0`` for an all-zero sparse matrix.
    """
    if sparse.issparse(matrix):
        return 0.0 if matrix.nnz == 0 else float(np.abs(matrix.data).max())
    return float(np.abs(matrix).max())


def check_operators(length: int) -> None:
    """Check every MPO against its independent sparse assembly.

    Args:
        length: Number of sites.
    """
    print(f"\nMPO versus independent sparse assembly, L={length}")
    for model in fx.MODELS:
        difference = fx.hamiltonian_mpo(length, model).to_sparse_matrix() - fx.sparse_hamiltonian(length, model)
        check(max_abs(difference) < ATOL, f"H[{model}] agrees ({max_abs(difference):.2e})")
    difference = fx.s2_mpo(length).to_sparse_matrix() - fx.sparse_s2(length)
    check(max_abs(difference) < ATOL, f"S^2 agrees ({max_abs(difference):.2e})")
    difference = fx.sz_mpo(length).to_sparse_matrix() - fx.sparse_sz(length)
    check(max_abs(difference) < ATOL, f"S^z agrees ({max_abs(difference):.2e})")


def check_symmetry(length: int) -> None:
    """Check the commutators the screen's model classification depends on.

    Args:
        length: Number of sites.
    """
    print(f"\nSymmetry structure, L={length}")
    s2 = fx.sparse_s2(length)
    sz = fx.sparse_sz(length)
    for model in fx.MODELS:
        ham = fx.sparse_hamiltonian(length, model)
        commutator = max_abs(s2 @ ham - ham @ s2)
        if fx.is_su2_symmetric(model):
            check(commutator < ATOL, f"[S^2, H[{model}]] = 0 ({commutator:.2e})")
        else:
            check(commutator > 1e-6, f"[S^2, H[{model}]] != 0 ({commutator:.2e})")
        commutator = max_abs(sz @ ham - ham @ sz)
        if model == "tfim":
            check(commutator > 1e-6, f"[S^z, H[{model}]] != 0 ({commutator:.2e})")
        else:
            check(commutator < ATOL, f"[S^z, H[{model}]] = 0 ({commutator:.2e})")


def check_spectrum(length: int) -> None:
    """Check the normalization of ``S^2`` against values known in closed form.

    Args:
        length: Number of sites.
    """
    print(f"\nKnown values of S^2, L={length}")
    s2 = np.asarray(fx.sparse_s2(length).toarray())

    # The eigenvalues of S^2 are s(s+1) over the half-integer or integer ladder up to L/2.
    eigenvalues = np.linalg.eigvalsh(s2)
    expected = {round(0.25 * total * (total + 2), 10) for total in range(length % 2, length + 1, 2)}
    observed = {round(float(value), 10) for value in eigenvalues}
    check(observed <= expected, f"spectrum lies on s(s+1) ({sorted(observed)})")

    # The fully polarized state is the highest-weight state of the maximal multiplet.
    polarized = np.zeros(1 << length, dtype=np.complex128)
    polarized[0] = 1.0
    value = float(np.vdot(polarized, s2 @ polarized).real)
    maximal = 0.25 * length * (length + 2)
    check(abs(value - maximal) < ATOL, f"<S^2> of the polarized state is {maximal} ({value:.12f})")

    # A Neel product state has <S^2> = 3L/4 + 2 sum_{i<j} s_i s_j with s alternating, which
    # collapses to L/2 for even L.
    neel = np.zeros(1 << length, dtype=np.complex128)
    neel[sum(1 << site for site in range(1, length, 2))] = 1.0
    value = float(np.vdot(neel, s2 @ neel).real)
    check(abs(value - length / 2) < ATOL, f"<S^2> of the Neel state is {length / 2} ({value:.12f})")


def check_states(length: int) -> None:
    """Check that every initial state is normalized and canonical.

    Args:
        length: Number of sites.
    """
    print(f"\nInitial states, L={length}")
    s2 = fx.sparse_s2(length)
    for name in fx.INITIAL_STATES:
        state = fx.initial_state(length, name)
        vector = state.to_vec()
        norm = float(np.vdot(vector, vector).real)
        value = float(np.vdot(vector, s2 @ vector).real / norm)
        check(abs(norm - 1.0) < 1e-10, f"{name}: normalized ({norm:.12f})")
        check(state.orthogonality_center == 0, f"{name}: centre at site 0")
        check(bool(np.all(np.isfinite(vector))), f"{name}: finite, <S^2> = {value:.6f}")


def check_manuscript_agreement() -> None:
    """Check the ``L = 16`` rebuilds against the manuscript's own operators."""
    print("\nAgreement with the manuscript at L=16")
    runner_path = HERE.parent / "l16_matched_optimized_2026-08-12" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("l16_runner", runner_path)
    if spec is None or spec.loader is None:
        check(False, f"could not load the manuscript runner at {runner_path}")
        return
    runner = importlib.util.module_from_spec(spec)
    sys.modules["l16_runner"] = runner
    spec.loader.exec_module(runner)

    length = runner.LENGTH
    for model, builder in (("tfim", runner.direct_ising_mpo), ("hs", runner.direct_haldane_shastry_mpo)):
        mine = fx.hamiltonian_mpo(length, model)
        theirs = builder()
        same_shapes = [a.shape == b.shape for a, b in zip(mine.tensors, theirs.tensors, strict=False)]
        check(all(same_shapes), f"{model}: tensor shapes match")
        if all(same_shapes):
            worst = max(float(np.abs(a - b).max()) for a, b in zip(mine.tensors, theirs.tensors, strict=False))
            check(worst < ATOL, f"{model}: tensors agree ({worst:.2e})")
        difference = mine.to_sparse_matrix() - runner.exact_sparse_hamiltonian(model)
        check(
            max_abs(difference) < ATOL,
            f"{model}: matches the manuscript sparse reference ({max_abs(difference):.2e})",
        )

    # The manuscript pads the Neel state for every non-TFIM model and |+>^L for the TFIM.
    for model, name in (("tfim", "plus"), ("hs", "neel")):
        offset = fx.SEED + (0 if model == "tfim" else 1) - fx.INITIAL_STATES.index(name)
        mine = fx.initial_state(length, name, seed=offset)
        theirs = runner.padded_initial_state(model)
        overlap = abs(complex(np.vdot(theirs.to_vec(), mine.to_vec())))
        check(abs(overlap - 1.0) < 1e-9, f"{model}: initial state matches the manuscript ({overlap:.12f})")


def main() -> int:
    """Run every check and report.

    Returns:
        ``0`` if all checks passed, ``1`` otherwise.
    """
    for length in (4, 6):
        check_operators(length)
        check_symmetry(length)
        check_spectrum(length)
        check_states(length)
    check_operators(16)
    check_manuscript_agreement()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} check(s) FAILED:")
        for message in _FAILURES:
            print(f"  - {message}")
        return 1
    print("All fixture checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
