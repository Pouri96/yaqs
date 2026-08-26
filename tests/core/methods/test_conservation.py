# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the post-compression energy correction."""

from __future__ import annotations

from copy import deepcopy
from itertools import starmap
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.analog.analog_tjm import capture_energy_target
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    DigitalSimParams,
    EvolutionMode,
)
from mqt.yaqs.core.methods.bug import (
    _postprocess_bug_state,  # ruff: ignore[import-private-name]  # placement inside the compression step is pinned here
    bug,
    bug_sweep,
)
from mqt.yaqs.core.methods.conservation import (
    energy_expectation,
    restore_energy_at_center,
    solve_min_modulus_root,
    validate_energy_conservation,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crandn(shape: tuple[int, ...], seed: int) -> NDArray[np.complex128]:
    """Draw a complex normal array of the given shape.

    Returns:
        Complex array with the requested shape.
    """
    rng = np.random.default_rng(seed)
    return np.asarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2), dtype=np.complex128)


def random_mps(shapes: list[tuple[int, int, int]], seed: int = 11) -> MPS:
    """Create a normalized random MPS canonical at site 0.

    Returns:
        A normalized :class:`MPS` with the center tracked at site ``0``.
    """
    mps = MPS(len(shapes), tensors=[crandn(shape, seed + i) for i, shape in enumerate(shapes)])
    mps.set_canonical_form(0, decomposition="QR")
    mps.set_center(0)
    mps.tensors[0] /= np.linalg.norm(mps.tensors[0])
    return mps


def sim_params(
    *,
    conserve_energy: bool,
    max_bond_dim: int | None,
    svd_threshold: float = 0.0,
    conserve_tol: float = 1e-12,
    dt: float = 0.05,
    elapsed_time: float = 0.5,
) -> AnalogSimParams:
    """Build analog parameters for a BUG run.

    Returns:
        Configured :class:`AnalogSimParams`.
    """
    return AnalogSimParams(
        elapsed_time=elapsed_time,
        dt=dt,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=max_bond_dim,
        svd_threshold=svd_threshold,
        krylov_tol=1e-12,
        conserve_energy=conserve_energy,
        conserve_tol=conserve_tol,
    )


def tensors_identical(left: MPS, right: MPS) -> bool:
    """Compare two MPS tensor lists for exact bitwise equality.

    Returns:
        True if every tensor has the same shape and identical entries.
    """
    return len(left.tensors) == len(right.tensors) and all(
        a.shape == b.shape and np.array_equal(a, b) for a, b in zip(left.tensors, right.tensors, strict=True)
    )


# --- Scalar root solve ---


def test_root_matches_reference_roots() -> None:
    """The solver returns the smaller-modulus real root of a well-conditioned quadratic."""
    a, b, c = 0.7, 3.0, -1.1
    expected = min(np.roots([a, b, c]), key=abs)
    assert solve_min_modulus_root(a, b, c) == pytest.approx(float(expected.real), rel=1e-13)


def test_root_accurate_when_quadratic_term_is_negligible() -> None:
    """With a tiny quadratic coefficient the root stays accurate against the exact value."""
    a, b, c = 1e-18, 2.0, -1e-13
    root = solve_min_modulus_root(a, b, c)
    assert root is not None
    assert a * root**2 + b * root + c == pytest.approx(0.0, abs=1e-28)


def test_root_degrades_to_the_linear_solution() -> None:
    """A vanishing quadratic coefficient reproduces the linear root ``-c / b``."""
    b, c = 4.0, -2.0
    assert solve_min_modulus_root(0.0, b, c) == pytest.approx(-c / b, rel=1e-15)


def test_root_reports_no_real_solution() -> None:
    """A negative discriminant returns ``None`` instead of a complex root."""
    assert solve_min_modulus_root(1.0, 1.0, 1.0) is None


def test_root_reports_degenerate_denominator() -> None:
    """An identically zero quadratic offers no displacement and returns ``None``."""
    assert solve_min_modulus_root(0.0, 0.0, 0.0) is None


# --- Expectation value ---


def test_energy_expectation_matches_dense_contraction() -> None:
    """The environment contraction agrees with the dense expectation value."""
    mps = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    vec = mps.to_vec()
    expected = np.vdot(vec, mpo.to_matrix_mps_order() @ vec).real / np.vdot(vec, vec).real
    assert energy_expectation(mps, mpo) == pytest.approx(float(expected), rel=1e-12)


def test_energy_expectation_is_gauge_independent() -> None:
    """Moving the orthogonality center does not change the expectation value."""
    mps = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    before = energy_expectation(mps, mpo)
    shifted = deepcopy(mps)
    shifted.set_canonical_form(2, decomposition="QR")
    assert energy_expectation(shifted, mpo) == pytest.approx(before, rel=1e-12)


def test_energy_expectation_length_mismatch_raises() -> None:
    """A Hamiltonian on a different number of sites is rejected."""
    with pytest.raises(ValueError, match="same number of sites"):
        energy_expectation(random_mps([(2, 1, 2), (2, 2, 1)]), MPO.ising(3, 1.0, 0.5))


# --- Correction at the center ---


def compressed_state(length: int, chi: int) -> tuple[MPS, MPO, float]:
    """Run one uncompressed half-sweep and compress it, leaving the center at the endpoint.

    Returns:
        The compressed state, the Hamiltonian, and the initial normalized energy.
    """
    shapes = [(2, 1, 4), *[(2, 4, 4)] * (length - 2), (2, 4, 1)]
    state = random_mps(shapes)
    mpo = MPO.ising(length, 1.0, 0.5)
    target = energy_expectation(state, mpo)
    bug_sweep(state, mpo, dt=0.05, krylov_tol=1e-12)
    _postprocess_bug_state(state, mpo, sim_params(conserve_energy=False, max_bond_dim=chi), normalize=False)
    return state, mpo, target


def test_restore_energy_pins_the_invariant_exactly() -> None:
    """After the correction the normalized energy equals the recorded target."""
    state, mpo, target = compressed_state(6, 2)
    assert restore_energy_at_center(state, mpo, target)
    assert energy_expectation(state, mpo) == pytest.approx(target, abs=1e-13)


def test_restore_energy_preserves_every_bond_dimension() -> None:
    """The correction changes no tensor shape."""
    state, mpo, target = compressed_state(6, 2)
    before = [tensor.shape for tensor in state.tensors]
    assert restore_energy_at_center(state, mpo, target)
    assert [tensor.shape for tensor in state.tensors] == before


def test_restore_energy_touches_only_the_center() -> None:
    """Every tensor other than the orthogonality center is left untouched."""
    state, mpo, target = compressed_state(6, 2)
    before = [tensor.copy() for tensor in state.tensors]
    center = state.orthogonality_center
    assert restore_energy_at_center(state, mpo, target)
    for site, tensor in enumerate(state.tensors):
        if site != center:
            assert np.array_equal(tensor, before[site])


def test_restore_energy_skips_inside_the_guard() -> None:
    """The correction is omitted when the invariant already holds."""
    state, mpo, _ = compressed_state(6, 2)
    before = [tensor.copy() for tensor in state.tensors]
    assert not restore_energy_at_center(state, mpo, energy_expectation(state, mpo))
    assert all(starmap(np.array_equal, zip(state.tensors, before, strict=True)))


def test_restore_energy_requires_a_known_gauge() -> None:
    """An MPS without a tracked center is rejected."""
    state, mpo, target = compressed_state(4, 2)
    state.set_center(None)
    with pytest.raises(ValueError, match="gauge unknown"):
        restore_energy_at_center(state, mpo, target)


def test_restore_energy_length_mismatch_raises() -> None:
    """A Hamiltonian on a different number of sites is rejected."""
    state, _, target = compressed_state(4, 2)
    with pytest.raises(ValueError, match="same number of sites"):
        restore_energy_at_center(state, MPO.ising(5, 1.0, 0.5), target)


def test_guard_is_relative_to_the_target_scale() -> None:
    """Scaling the Hamiltonian scales the guard, so the skip decision is unchanged."""
    scale = 1e6
    state, mpo, _ = compressed_state(6, 2)
    scaled = deepcopy(mpo)
    # Scale one tensor only: an MPO is a product over sites, so scaling all of them
    # would multiply the operator by ``scale ** length``.
    scaled.tensors[0] = scale * scaled.tensors[0]

    plain = energy_expectation(state, mpo)
    # A drift of 1e-9 relative to the target sits above a 1e-12 relative guard at both
    # scales, and below an absolute 1e-12 guard at neither: the correction must fire twice.
    assert restore_energy_at_center(deepcopy(state), mpo, plain * (1 + 1e-9))
    assert restore_energy_at_center(deepcopy(state), scaled, plain * scale * (1 + 1e-9))

    # A drift far below the relative guard must be skipped at both scales.
    assert not restore_energy_at_center(deepcopy(state), mpo, plain * (1 + 1e-15))
    assert not restore_energy_at_center(deepcopy(state), scaled, plain * scale * (1 + 1e-15))


# --- Integration with the BUG step ---


def test_flag_off_leaves_the_step_bit_identical() -> None:
    """With ``conserve_energy`` off a supplied target changes nothing."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    target = energy_expectation(reference, mpo)
    plain = deepcopy(reference)
    with_target = deepcopy(reference)
    for _ in range(4):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2))
        bug(with_target, mpo, sim_params(conserve_energy=False, max_bond_dim=2), energy_target=target)
    assert tensors_identical(plain, with_target)


def test_missing_target_leaves_the_step_bit_identical() -> None:
    """With the flag on but no target the step is the uncorrected one."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    plain = deepcopy(reference)
    armed = deepcopy(reference)
    for _ in range(4):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2))
        bug(armed, mpo, sim_params(conserve_energy=True, max_bond_dim=2))
    assert tensors_identical(plain, armed)


def test_uncompressed_run_is_left_bit_identical() -> None:
    """A step that discards no weight is untouched by the correction."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    target = energy_expectation(reference, mpo)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    for _ in range(8):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=None))
        bug(corrected, mpo, sim_params(conserve_energy=True, max_bond_dim=None), energy_target=target)
    assert tensors_identical(plain, corrected)


def test_corrected_run_conserves_energy_at_fixed_rank() -> None:
    """A truncating run keeps the energy at round-off with the bond profile unchanged."""
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    target = energy_expectation(reference, mpo)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    plain_bonds, corrected_bonds = [], []
    for _ in range(10):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2))
        bug(corrected, mpo, sim_params(conserve_energy=True, max_bond_dim=2), energy_target=target)
        plain_bonds.append(plain.bond_dimensions())
        corrected_bonds.append(corrected.bond_dimensions())

    uncorrected_drift = abs(energy_expectation(plain, mpo) - target)
    corrected_drift = abs(energy_expectation(corrected, mpo) - target)
    assert uncorrected_drift > 1e-9, "the reference run must actually drift for this test to bite"
    assert corrected_drift < 1e-13
    assert plain_bonds == corrected_bonds


def test_correction_runs_in_both_half_sweeps_and_reflected_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of the two compressions is corrected, the second with the reflected MPO."""
    seen: list[NDArray[np.complex128]] = []
    original = restore_energy_at_center

    def spy(state: MPS, mpo: MPO, target: float, *, tol: float) -> bool:
        seen.append(mpo.tensors[0].copy())
        return original(state, mpo, target, tol=tol)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug.restore_energy_at_center", spy)
    state = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    bug(state, mpo, sim_params(conserve_energy=True, max_bond_dim=2), energy_target=energy_expectation(state, mpo))

    assert len(seen) == 2
    assert np.array_equal(seen[0], mpo.tensors[0])
    assert np.array_equal(seen[1], mpo.reflected().tensors[0])


def test_correction_adds_to_the_compression_error_orthogonally() -> None:
    """The Pythagorean identity of the no-error-reduction clause holds to round-off."""
    state = random_mps([(2, 1, 4), *[(2, 4, 4)] * 3, (2, 4, 1)])
    mpo = MPO.ising(5, 1.0, 0.5)
    target = energy_expectation(state, mpo)

    bug_sweep(state, mpo, dt=0.05, krylov_tol=1e-12)
    uncompressed = state.to_vec().copy()

    compressed = deepcopy(state)
    _postprocess_bug_state(compressed, mpo, sim_params(conserve_energy=False, max_bond_dim=2), normalize=False)
    corrected = deepcopy(state)
    _postprocess_bug_state(
        corrected,
        mpo,
        sim_params(conserve_energy=True, max_bond_dim=2),
        normalize=False,
        energy_target=target,
    )

    compressed_vec = compressed.to_vec()
    corrected_vec = corrected.to_vec()
    displacement = float(np.linalg.norm(corrected_vec - compressed_vec) ** 2)
    assert displacement > 0.0, "the correction must move the state for this test to bite"

    truncation = float(np.linalg.norm(compressed_vec - uncompressed) ** 2)
    total = float(np.linalg.norm(corrected_vec - uncompressed) ** 2)
    assert total == pytest.approx(truncation + displacement, rel=1e-12)
    assert total > truncation


# --- Configuration surface ---


def test_conserve_energy_defaults_to_off() -> None:
    """The feature is opt-in on both parameter classes."""
    assert AnalogSimParams(elapsed_time=0.1, dt=0.1).conserve_energy is False
    assert AnalogSimParams(elapsed_time=0.1, dt=0.1).conserve_tol == pytest.approx(1e-13)
    assert DigitalSimParams().conserve_tol == pytest.approx(1e-13)


def test_default_guard_leaves_an_uncompressed_run_bit_identical() -> None:
    """At the default tolerance the guard stays shut when nothing is discarded.

    This pins the choice of default: the guard must not open on the local solver's
    residual alone, or an exact step stops being a no-op and the compatibility with
    the uncompressed conservation statement is lost. Short chains are the binding
    case, since their residual is largest relative to the guard.
    """
    for shapes in ([(2, 1, 4), (2, 4, 4), (2, 4, 1)], [(2, 1, 4), (2, 4, 8), (2, 8, 4), (2, 4, 1)]):
        reference = random_mps(shapes)
        mpo = MPO.ising(len(shapes), 1.0, 0.5)
        target = energy_expectation(reference, mpo)
        plain = deepcopy(reference)
        corrected = deepcopy(reference)
        default_tol = AnalogSimParams(elapsed_time=0.1, dt=0.1).conserve_tol
        off = sim_params(conserve_energy=False, max_bond_dim=None, conserve_tol=default_tol)
        on = sim_params(conserve_energy=True, max_bond_dim=None, conserve_tol=default_tol)
        for _ in range(10):
            bug(plain, mpo, off)
            bug(corrected, mpo, on, energy_target=target)
        assert tensors_identical(plain, corrected)


def test_conserve_tol_rejects_invalid_values() -> None:
    """A negative or non-finite tolerance is rejected at construction."""
    with pytest.raises(ValueError, match="conserve_tol"):
        AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_tol=-1.0)
    with pytest.raises(ValueError, match="conserve_tol"):
        AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_tol=float("nan"))


def test_validate_rejects_energy_conservation_under_noise() -> None:
    """The energy is not an invariant of a noisy trajectory, so the combination is refused."""
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    validate_energy_conservation(params, has_noise=False)
    with pytest.raises(ValueError, match="noise model"):
        validate_energy_conservation(params, has_noise=True)


def test_capture_target_returns_none_when_the_flag_is_off() -> None:
    """No target is recorded, and no contraction is run, with the feature off."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = sim_params(conserve_energy=False, max_bond_dim=2)
    assert capture_energy_target(state, MPO.ising(2, 1.0, 0.5), None, params) is None


def test_capture_target_matches_the_initial_expectation_value() -> None:
    """The recorded target is the normalized initial energy of the evolved state."""
    state = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    captured = capture_energy_target(state, mpo, None, params)
    assert captured == pytest.approx(energy_expectation(state, mpo), rel=1e-14)


def test_capture_target_requires_the_bug_evolution_mode() -> None:
    """Requesting the correction under TDVP is an error rather than a silent no-op."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        evolution_mode=EvolutionMode.TDVP,
        conserve_energy=True,
    )
    with pytest.raises(ValueError, match="requires evolution_mode"):
        capture_energy_target(state, MPO.ising(2, 1.0, 0.5), None, params)


def test_capture_target_rejects_a_noise_model() -> None:
    """The driver refuses the correction on an open-system run."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="noise model"):
        capture_energy_target(state, MPO.ising(2, 1.0, 0.5), noise, params)
