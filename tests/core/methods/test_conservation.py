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

from mqt.yaqs.analog.analog_tjm import capture_conservation_target
from mqt.yaqs.analog.evolution import apply_unitary_evolution
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
    EnergyTarget,
    _solve_min_modulus_root,  # ruff: ignore[import-private-name]  # module-private numeric kernel
    apply_conservation,
    expectation_value,
    restore_invariant_at_center,
    validate_conservation,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.simulation_parameters import TDVPMode


# The parameter default, so the direct-kernel tests track whatever it is set to.
DEFAULT_TOL = AnalogSimParams(elapsed_time=0.1, dt=0.1).conserve_tol


def armed(value: float, tol: float = DEFAULT_TOL) -> EnergyTarget:
    """Return the correction settings that pin ``value``.

    Returns:
        The target and its tolerance, as the integrators receive them.
    """
    return EnergyTarget(value, tol)


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
    conserve_tol: float = DEFAULT_TOL,
    dt: float = 0.05,
    elapsed_time: float = 0.5,
    evolution_mode: EvolutionMode = EvolutionMode.BUG,
    tdvp_mode: TDVPMode = "2site",
) -> AnalogSimParams:
    """Build analog parameters for one evolution mode.

    Returns:
        Configured :class:`AnalogSimParams`.
    """
    return AnalogSimParams(
        elapsed_time=elapsed_time,
        dt=dt,
        evolution_mode=evolution_mode,
        max_bond_dim=max_bond_dim,
        svd_threshold=svd_threshold,
        krylov_tol=1e-12,
        conserve_energy=conserve_energy,
        conserve_tol=conserve_tol,
        tdvp_mode=tdvp_mode,
    )


def run_steps(state: MPS, mpo: MPO, params: AnalogSimParams, steps: int) -> None:
    """Run ``steps`` unitary steps through the production dispatcher.

    The target comes from :func:`capture_conservation_target`, so the tests take the same
    path a simulation does.
    """
    target = capture_conservation_target(state, mpo, None, params)
    for _ in range(steps):
        apply_unitary_evolution(state, mpo, params, conservation_target=target)


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
    assert _solve_min_modulus_root(a, b, c) == pytest.approx(float(expected.real), rel=1e-13)


def test_root_accurate_when_quadratic_term_is_negligible() -> None:
    """With a tiny quadratic coefficient the root stays accurate against the exact value."""
    a, b, c = 1e-18, 2.0, -1e-13
    root = _solve_min_modulus_root(a, b, c)
    assert root is not None
    assert a * root**2 + b * root + c == pytest.approx(0.0, abs=1e-28)


def test_root_degrades_to_the_linear_solution() -> None:
    """A vanishing quadratic coefficient reproduces the linear root ``-c / b``."""
    b, c = 4.0, -2.0
    assert _solve_min_modulus_root(0.0, b, c) == pytest.approx(-c / b, rel=1e-15)


def test_root_reports_no_real_solution() -> None:
    """A negative discriminant returns ``None`` instead of a complex root."""
    assert _solve_min_modulus_root(1.0, 1.0, 1.0) is None


def test_root_reports_degenerate_denominator() -> None:
    """An identically zero quadratic offers no displacement and returns ``None``."""
    assert _solve_min_modulus_root(0.0, 0.0, 0.0) is None


# --- Expectation value ---


def test_expectation_value_matches_dense_contraction() -> None:
    """The environment contraction agrees with the dense expectation value."""
    mps = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    vec = mps.to_vec()
    expected = np.vdot(vec, mpo.to_matrix_mps_order() @ vec).real / np.vdot(vec, vec).real
    assert expectation_value(mps, mpo) == pytest.approx(float(expected), rel=1e-12)


def test_expectation_value_is_gauge_independent() -> None:
    """Moving the orthogonality center does not change the expectation value."""
    mps = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    before = expectation_value(mps, mpo)
    shifted = deepcopy(mps)
    shifted.set_canonical_form(2, decomposition="QR")
    assert expectation_value(shifted, mpo) == pytest.approx(before, rel=1e-12)


def test_expectation_value_length_mismatch_raises() -> None:
    """A Hamiltonian on a different number of sites is rejected."""
    with pytest.raises(ValueError, match="same number of sites"):
        expectation_value(random_mps([(2, 1, 2), (2, 2, 1)]), MPO.ising(3, 1.0, 0.5))


# --- Correction at the center ---


def compressed_state(length: int, chi: int) -> tuple[MPS, MPO, float]:
    """Run one uncompressed half-sweep and compress it, leaving the center at the endpoint.

    Returns:
        The compressed state, the Hamiltonian, and the initial normalized energy.
    """
    shapes = [(2, 1, 4), *[(2, 4, 4)] * (length - 2), (2, 4, 1)]
    state = random_mps(shapes)
    mpo = MPO.ising(length, 1.0, 0.5)
    target = expectation_value(state, mpo)
    bug_sweep(state, mpo, dt=0.05, krylov_tol=1e-12)
    _postprocess_bug_state(state, mpo, sim_params(conserve_energy=False, max_bond_dim=chi), normalize=False)
    return state, mpo, target


def test_restore_energy_pins_the_invariant_exactly() -> None:
    """After the correction the normalized energy equals the target."""
    state, mpo, target = compressed_state(6, 2)
    assert restore_invariant_at_center(state, mpo, target, tol=DEFAULT_TOL)
    assert expectation_value(state, mpo) == pytest.approx(target, abs=1e-13)


def test_restore_energy_preserves_every_bond_dimension() -> None:
    """The correction changes no tensor shape."""
    state, mpo, target = compressed_state(6, 2)
    before = [tensor.shape for tensor in state.tensors]
    assert restore_invariant_at_center(state, mpo, target, tol=DEFAULT_TOL)
    assert [tensor.shape for tensor in state.tensors] == before


def test_restore_energy_touches_only_the_center() -> None:
    """Every tensor other than the orthogonality center is left untouched."""
    state, mpo, target = compressed_state(6, 2)
    before = [tensor.copy() for tensor in state.tensors]
    center = state.orthogonality_center
    assert restore_invariant_at_center(state, mpo, target, tol=DEFAULT_TOL)
    for site, tensor in enumerate(state.tensors):
        if site != center:
            assert np.array_equal(tensor, before[site])


def test_restore_energy_skips_inside_the_tolerance() -> None:
    """The correction is omitted when the energy already matches the target."""
    state, mpo, _ = compressed_state(6, 2)
    before = [tensor.copy() for tensor in state.tensors]
    assert not restore_invariant_at_center(state, mpo, expectation_value(state, mpo), tol=DEFAULT_TOL)
    assert all(starmap(np.array_equal, zip(state.tensors, before, strict=True)))


def test_restore_energy_requires_a_known_gauge() -> None:
    """An MPS without a tracked center is rejected."""
    state, mpo, target = compressed_state(4, 2)
    state.set_center(None)
    with pytest.raises(ValueError, match="gauge unknown"):
        restore_invariant_at_center(state, mpo, target, tol=DEFAULT_TOL)


def test_restore_energy_length_mismatch_raises() -> None:
    """A Hamiltonian on a different number of sites is rejected."""
    state, _, target = compressed_state(4, 2)
    with pytest.raises(ValueError, match="same number of sites"):
        restore_invariant_at_center(state, MPO.ising(5, 1.0, 0.5), target, tol=DEFAULT_TOL)


def test_tolerance_is_relative_to_the_target_scale() -> None:
    """Scaling the Hamiltonian scales the tolerance, so the skip decision is unchanged."""
    scale = 1e6
    state, mpo, _ = compressed_state(6, 2)
    scaled = deepcopy(mpo)
    # Scale one tensor only: an MPO is a product over sites, so scaling all of them
    # would multiply the operator by ``scale ** length``.
    scaled.tensors[0] = scale * scaled.tensors[0]

    plain = expectation_value(state, mpo)
    # A drift of 1e-9 relative to the target sits above a 1e-12 relative guard at both
    # scales, and below an absolute 1e-12 guard at neither: the correction must fire twice.
    assert restore_invariant_at_center(deepcopy(state), mpo, plain * (1 + 1e-9), tol=1e-12)
    assert restore_invariant_at_center(deepcopy(state), scaled, plain * scale * (1 + 1e-9), tol=1e-12)

    # A drift far below the relative guard must be skipped at both scales.
    assert not restore_invariant_at_center(deepcopy(state), mpo, plain * (1 + 1e-15), tol=1e-12)
    assert not restore_invariant_at_center(deepcopy(state), scaled, plain * scale * (1 + 1e-15), tol=1e-12)


# --- Integration with the BUG step ---


@pytest.mark.parametrize("evolution_mode", [EvolutionMode.BUG, EvolutionMode.TDVP])
def test_flag_off_never_enters_the_correction(evolution_mode: EvolutionMode, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``conserve_energy`` off the correction is never called.

    Both call sites are replaced by a function that fails, so this checks control
    flow rather than comparing two states that could agree by accident.
    """

    def forbidden(*_args: object, **_kwargs: object) -> bool:
        msg = "the correction ran on a run with conserve_energy off"
        raise AssertionError(msg)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug.apply_conservation", forbidden)
    monkeypatch.setattr("mqt.yaqs.analog.evolution.apply_conservation", forbidden)
    state = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    params = sim_params(conserve_energy=False, max_bond_dim=2, evolution_mode=evolution_mode)
    run_steps(state, MPO.ising(4, 1.0, 0.5), params, 4)


def test_missing_target_leaves_the_step_bit_identical() -> None:
    """A step given no target is the uncorrected step."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    plain = deepcopy(reference)
    armed = deepcopy(reference)
    for _ in range(4):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2))
        bug(armed, mpo, sim_params(conserve_energy=True, max_bond_dim=2))
    assert tensors_identical(plain, armed)


def test_uncompressed_run_is_left_bit_identical() -> None:
    """A step that truncates nothing is untouched by the correction."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    target = expectation_value(reference, mpo)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    for _ in range(8):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=None))
        bug(
            corrected,
            mpo,
            sim_params(conserve_energy=True, max_bond_dim=None),
            conservation_target=armed(target),
        )
    assert tensors_identical(plain, corrected)


def test_corrected_run_conserves_energy_at_fixed_rank() -> None:
    """A truncating run keeps the energy at round-off with the bond profile unchanged."""
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    target = expectation_value(reference, mpo)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    plain_bonds, corrected_bonds = [], []
    for _ in range(10):
        bug(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2))
        bug(
            corrected,
            mpo,
            sim_params(conserve_energy=True, max_bond_dim=2),
            conservation_target=armed(target),
        )
        plain_bonds.append(plain.bond_dimensions())
        corrected_bonds.append(corrected.bond_dimensions())

    uncorrected_drift = abs(expectation_value(plain, mpo) - target)
    corrected_drift = abs(expectation_value(corrected, mpo) - target)
    assert uncorrected_drift > 1e-9, "the reference run must actually drift for this test to bite"
    assert corrected_drift < 1e-13
    assert plain_bonds == corrected_bonds


def test_correction_runs_in_both_half_sweeps_and_reflected_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of the two compressions is corrected, the second with the reflected MPO."""
    seen: list[NDArray[np.complex128]] = []
    original = apply_conservation

    def spy(state: MPS, mpo: MPO, target: EnergyTarget) -> bool:
        seen.append(mpo.tensors[0].copy())
        return original(state, mpo, target)

    monkeypatch.setattr("mqt.yaqs.core.methods.bug.apply_conservation", spy)
    state = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    bug(
        state,
        mpo,
        sim_params(conserve_energy=True, max_bond_dim=2),
        conservation_target=armed(expectation_value(state, mpo)),
    )

    assert len(seen) == 2
    assert np.array_equal(seen[0], mpo.tensors[0])
    assert np.array_equal(seen[1], mpo.reflected().tensors[0])


def test_correction_adds_to_the_compression_error_orthogonally() -> None:
    """The correction adds to the compression error rather than reducing it."""
    state = random_mps([(2, 1, 4), *[(2, 4, 4)] * 3, (2, 4, 1)])
    mpo = MPO.ising(5, 1.0, 0.5)
    target = expectation_value(state, mpo)

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
        conservation_target=armed(target),
    )

    compressed_vec = compressed.to_vec()
    corrected_vec = corrected.to_vec()
    displacement = float(np.linalg.norm(corrected_vec - compressed_vec) ** 2)
    assert displacement > 0.0, "the correction must move the state for this test to bite"

    truncation = float(np.linalg.norm(compressed_vec - uncompressed) ** 2)
    total = float(np.linalg.norm(corrected_vec - uncompressed) ** 2)
    assert total == pytest.approx(truncation + displacement, rel=1e-12)
    assert total > truncation


# --- Integration with the TDVP step ---


@pytest.mark.parametrize("tdvp_mode", ["1site", "2site", "dynamic"])
def test_tdvp_leaves_one_center_at_site_zero(tdvp_mode: TDVPMode) -> None:
    """Every TDVP sweep geometry ends with the center at site 0.

    The correction is applied there, so this must hold for the 1-site, 2-site and
    dynamic sweeps alike.
    """
    state = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    params = sim_params(conserve_energy=False, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP, tdvp_mode=tdvp_mode)
    for _ in range(3):
        apply_unitary_evolution(state, MPO.ising(6, 1.0, 0.5), params)
        assert state.orthogonality_center == 0


def test_corrected_tdvp_run_conserves_energy_at_fixed_rank() -> None:
    """A truncating 2TDVP run keeps the energy at round-off with the bond profile unchanged."""
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    target = expectation_value(reference, mpo)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    off = sim_params(conserve_energy=False, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP)
    on = sim_params(conserve_energy=True, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP)
    run_steps(plain, mpo, off, 10)
    run_steps(corrected, mpo, on, 10)

    assert abs(expectation_value(plain, mpo) - target) > 1e-9, (
        "the reference run must actually drift for this test to bite"
    )
    assert abs(expectation_value(corrected, mpo) - target) < 1e-13
    assert corrected.bond_dimensions() == plain.bond_dimensions()


def test_tdvp_correction_touches_only_the_center_tensor() -> None:
    """One corrected step differs from the uncorrected step at site 0 alone."""
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    run_steps(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP), 1)
    run_steps(corrected, mpo, sim_params(conserve_energy=True, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP), 1)

    assert not np.array_equal(corrected.tensors[0], plain.tensors[0])
    for site in range(1, 6):
        assert np.array_equal(corrected.tensors[site], plain.tensors[site])


def test_tdvp_correction_leaves_norm_handling_to_the_integrator() -> None:
    """The correction does not renormalize a TDVP step.

    TDVP leaves the norm slightly off 1 after truncation, and the correction must not
    silently fix that on its behalf.
    """
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    run_steps(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP), 10)
    run_steps(corrected, mpo, sim_params(conserve_energy=True, max_bond_dim=2, evolution_mode=EvolutionMode.TDVP), 10)

    plain_drift = abs(float(plain.scalar_product(plain).real) - 1.0)
    corrected_drift = abs(float(corrected.scalar_product(corrected).real) - 1.0)
    assert plain_drift > 1e-6, "TDVP must leave the norm off 1 for this test to bite"
    assert corrected_drift == pytest.approx(plain_drift, rel=1e-3)


def test_uncompressed_tdvp_run_is_left_bit_identical() -> None:
    """1TDVP performs no truncation, so the corrected run is identical."""
    reference = random_mps([(2, 1, 4), (2, 4, 4), (2, 4, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    plain = deepcopy(reference)
    corrected = deepcopy(reference)
    off = sim_params(conserve_energy=False, max_bond_dim=None, evolution_mode=EvolutionMode.TDVP, tdvp_mode="1site")
    on = sim_params(conserve_energy=True, max_bond_dim=None, evolution_mode=EvolutionMode.TDVP, tdvp_mode="1site")
    run_steps(plain, mpo, off, 8)
    run_steps(corrected, mpo, on, 8)
    assert tensors_identical(plain, corrected)


# --- Configuration surface ---


def test_conserve_energy_defaults_to_off() -> None:
    """The feature is opt-in."""
    assert AnalogSimParams(elapsed_time=0.1, dt=0.1).conserve_energy is False


def test_conserve_tol_rejects_invalid_values() -> None:
    """A negative or non-finite tolerance is rejected at construction."""
    with pytest.raises(ValueError, match="conserve_tol"):
        AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_tol=-1.0)
    with pytest.raises(ValueError, match="conserve_tol"):
        AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_tol=float("nan"))


def test_conserve_tol_reaches_the_correction() -> None:
    """``conserve_tol`` is applied by the correction, not merely stored.

    A tolerance wide enough to cover the whole drift switches the correction off,
    which is only visible if the configured value is the one used.
    """
    reference = random_mps([(2, 1, 4), *[(2, 4, 4)] * 4, (2, 4, 1)])
    mpo = MPO.ising(6, 1.0, 0.5)
    plain = deepcopy(reference)
    wide = deepcopy(reference)
    run_steps(plain, mpo, sim_params(conserve_energy=False, max_bond_dim=2), 6)
    run_steps(wide, mpo, sim_params(conserve_energy=True, max_bond_dim=2, conserve_tol=1.0), 6)
    assert tensors_identical(plain, wide)


def test_digital_parameters_expose_no_conservation_setting() -> None:
    """Digital simulation has no energy conservation settings.

    Circuits evolve under gate generators rather than a Hamiltonian, so the settings
    are absent rather than present and silently ignored.
    """
    assert not hasattr(DigitalSimParams(), "conserve_energy")
    with pytest.raises(TypeError, match="conserve_energy"):
        DigitalSimParams(conserve_energy=True)  # ty: ignore[unknown-argument]


def test_default_conserve_tol_leaves_an_uncompressed_run_bit_identical() -> None:
    """At the default tolerance a run that truncates nothing is left unchanged.

    This pins the choice of default: it must not be so tight that the local solver's
    own residual triggers the correction. Short chains are the binding case, since
    their residual is largest relative to the tolerance.
    """
    for shapes in ([(2, 1, 4), (2, 4, 4), (2, 4, 1)], [(2, 1, 4), (2, 4, 8), (2, 8, 4), (2, 4, 1)]):
        reference = random_mps(shapes)
        mpo = MPO.ising(len(shapes), 1.0, 0.5)
        target = expectation_value(reference, mpo)
        plain = deepcopy(reference)
        corrected = deepcopy(reference)
        off = sim_params(conserve_energy=False, max_bond_dim=None)
        on = sim_params(conserve_energy=True, max_bond_dim=None)
        for _ in range(10):
            bug(plain, mpo, off)
            bug(corrected, mpo, on, conservation_target=armed(target))
        assert tensors_identical(plain, corrected)


def test_validate_rejects_energy_conservation_under_noise() -> None:
    """Energy conservation and a noise model are refused together."""
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    validate_conservation(params, has_noise=False)
    with pytest.raises(ValueError, match="noise model"):
        validate_conservation(params, has_noise=True)


def test_capture_target_returns_none_when_the_flag_is_off() -> None:
    """No target is recorded, and no contraction is run, with the feature off."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = sim_params(conserve_energy=False, max_bond_dim=2)
    assert capture_conservation_target(state, MPO.ising(2, 1.0, 0.5), None, params) is None


def test_capture_target_matches_the_initial_expectation_value() -> None:
    """The recorded target is the normalized initial energy of the evolved state."""
    state = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    captured = capture_conservation_target(state, mpo, None, params)
    assert captured is not None
    assert captured.value == pytest.approx(expectation_value(state, mpo), rel=1e-14)


@pytest.mark.parametrize("evolution_mode", [EvolutionMode.BUG, EvolutionMode.TDVP])
def test_capture_target_serves_either_evolution_mode(evolution_mode: EvolutionMode) -> None:
    """Both TDVP and BUG receive a target."""
    state = random_mps([(2, 1, 3), (2, 3, 3), (2, 3, 1)])
    mpo = MPO.ising(3, 1.0, 0.5)
    params = sim_params(conserve_energy=True, max_bond_dim=2, evolution_mode=evolution_mode)
    captured = capture_conservation_target(state, mpo, None, params)
    assert captured is not None
    assert captured.value == pytest.approx(expectation_value(state, mpo), rel=1e-14)
    assert captured.tol == params.conserve_tol


def test_capture_target_rejects_a_noise_model() -> None:
    """The driver refuses the correction on an open-system run."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = sim_params(conserve_energy=True, max_bond_dim=2)
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="noise model"):
        capture_conservation_target(state, MPO.ising(2, 1.0, 0.5), noise, params)
