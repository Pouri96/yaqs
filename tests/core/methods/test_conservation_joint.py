# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the multi-observable post-compression correction.

The energy-only path is covered by ``test_conservation.py`` and must stay bit-identical; these
tests cover what was added around it. The three properties worth pinning down are that the
generalized scalar solve is the same routine the energy path already used, that the joint
solve restores several observables at once where a sequence of scalar solves cannot, and that
its acceptance test is monotone in exactly the quantity it tests and not in any other.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.analog.analog_tjm import capture_conservation_target
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods.bug import bug
from mqt.yaqs.core.methods.conservation import (
    ConservationTargets,
    EnergyTarget,
    apply_conservation,
    expectation_value,
    record_conservation_targets,
    restore_invariant_at_center,
    restore_invariants_joint,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

TOL = 1e-13


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


def spin_mpos(length: int) -> tuple[MPO, MPO]:
    """Build a total-spin and a total-magnetization MPO.

    Two observables that do not commute with each other's effective operator in general, which
    is the case the joint solve exists for.

    Returns:
        The ``S^2`` and ``S^z`` MPOs on ``length`` sites.
    """
    identity = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2

    s2_bulk = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    s2_bulk[0, :, :, 0] = identity
    s2_bulk[0, :, :, 4] = 0.75 * identity
    for axis, operator in enumerate((sx, sy, sz), start=1):
        s2_bulk[0, :, :, axis] = operator
        s2_bulk[axis, :, :, axis] = identity
        s2_bulk[axis, :, :, 4] = 2.0 * operator
    s2_bulk[4, :, :, 4] = identity

    sz_bulk = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    sz_bulk[0, :, :, 0] = identity
    sz_bulk[0, :, :, 1] = sz
    sz_bulk[1, :, :, 1] = identity

    def build(bulk: NDArray[np.complex128]) -> MPO:
        blocks = [bulk[0:1], *(bulk.copy() for _ in range(length - 2)), bulk[:, :, :, -1:]]
        mpo = MPO()
        mpo.custom([np.transpose(block, (1, 2, 0, 3)).copy() for block in blocks], transpose=False)
        return mpo

    return build(s2_bulk), build(sz_bulk)


def heisenberg_mpo(length: int, delta: float) -> MPO:
    """Build the bond-5 XXZ MPO.

    ``delta = 1`` is the isotropic point, which commutes with ``S^2``; any ``delta`` commutes
    with ``S^z``.

    Returns:
        The Hamiltonian as an :class:`MPO` on ``length`` sites.
    """
    identity = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2

    bulk = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    bulk[0, :, :, 0] = identity
    bulk[0, :, :, 1], bulk[0, :, :, 2], bulk[0, :, :, 3] = sx, sy, sz
    bulk[1, :, :, 4], bulk[2, :, :, 4], bulk[3, :, :, 4] = sx, sy, delta * sz
    bulk[4, :, :, 4] = identity

    blocks = [bulk[0:1], *(bulk.copy() for _ in range(length - 2)), bulk[:, :, :, -1:]]
    mpo = MPO()
    mpo.custom([np.transpose(block, (1, 2, 0, 3)).copy() for block in blocks], transpose=False)
    return mpo


SHAPES = [(2, 1, 4), (2, 4, 4), (2, 4, 4), (2, 4, 1)]


# --- The generalized scalar solve ---


def test_generalized_solve_reproduces_the_energy_solve() -> None:
    """``restore_invariant_at_center`` is the routine the energy path already ran."""
    state = random_mps(SHAPES)
    mpo = MPO.ising(4, 1.0, 0.5)
    target = expectation_value(state, mpo) * (1 + 1e-6)

    through_energy = deepcopy(state)
    through_general = deepcopy(state)
    assert restore_invariant_at_center(through_energy, mpo, target, tol=TOL)
    assert restore_invariant_at_center(through_general, mpo, target, tol=TOL)
    for left, right in zip(through_energy.tensors, through_general.tensors, strict=True):
        assert np.array_equal(left, right)


def test_scalar_solve_restores_a_non_energy_observable() -> None:
    """The scalar solve pins any Hermitian observable, not only the Hamiltonian."""
    state = random_mps(SHAPES)
    s2, _ = spin_mpos(4)
    target = expectation_value(state, s2) * (1 + 1e-4)

    assert restore_invariant_at_center(state, s2, target, tol=TOL)
    assert expectation_value(state, s2) == pytest.approx(target, abs=1e-12)


def test_scalar_solve_preserves_every_bond_dimension() -> None:
    """The displacement is confined to the center tensor, so no bond changes."""
    state = random_mps(SHAPES)
    s2, _ = spin_mpos(4)
    before = state.bond_dimensions()

    assert restore_invariant_at_center(state, s2, expectation_value(state, s2) * (1 + 1e-4), tol=TOL)
    assert state.bond_dimensions() == before


# --- The joint solve ---


def test_sequential_solves_cannot_hold_both_observables() -> None:
    """Applying the scalar solve once per observable leaves the earlier one perturbed.

    This is the failure the joint solve exists to remove, so it is worth pinning down rather
    than assuming: the second displacement moves the first observable off its target.
    """
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    targets = (expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3)

    sequential = deepcopy(state)
    restore_invariant_at_center(sequential, s2, targets[0], tol=TOL)
    restore_invariant_at_center(sequential, sz, targets[1], tol=TOL)

    # The last observable applied is exact; the first has been pushed off.
    assert expectation_value(sequential, sz) == pytest.approx(targets[1], abs=1e-12)
    assert abs(expectation_value(sequential, s2) - targets[0]) > 1e-9


def test_joint_solve_restores_both_observables() -> None:
    """One displacement in the span of the gradients meets both constraints at once."""
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    targets = (expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3)

    assert restore_invariants_joint(state, [s2, sz], targets, tol=TOL)
    assert expectation_value(state, s2) == pytest.approx(targets[0], abs=1e-10)
    assert expectation_value(state, sz) == pytest.approx(targets[1], abs=1e-10)


def test_joint_solve_preserves_every_bond_dimension() -> None:
    """The joint displacement is also confined to the center tensor."""
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    before = state.bond_dimensions()
    targets = (expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3)

    assert restore_invariants_joint(state, [s2, sz], targets, tol=TOL)
    assert state.bond_dimensions() == before


def test_joint_solve_touches_only_the_center() -> None:
    """Every tensor away from the orthogonality center is left bitwise unchanged."""
    state = random_mps(SHAPES)
    reference = deepcopy(state)
    s2, sz = spin_mpos(4)
    targets = (expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3)

    assert restore_invariants_joint(state, [s2, sz], targets, tol=TOL)
    assert not np.array_equal(state.tensors[0], reference.tensors[0])
    for site in range(1, state.length):
        assert np.array_equal(state.tensors[site], reference.tensors[site])


def test_joint_solve_never_increases_the_worst_residual() -> None:
    """The line search is monotone in ``max_a |<O_a> - target_a|``.

    This is the only guarantee the acceptance test provides, and the reason the step is damped
    at all: an undamped Newton step overshoots on an ill-conditioned covariance.
    """
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    targets = (expectation_value(state, s2) * (1 + 1e-2), expectation_value(state, sz) + 1e-2)

    def worst(current: MPS) -> float:
        return max(
            abs(expectation_value(current, s2) - targets[0]),
            abs(expectation_value(current, sz) - targets[1]),
        )

    before = worst(state)
    restore_invariants_joint(state, [s2, sz], targets, tol=TOL)
    assert worst(state) <= before


def test_joint_solve_skips_when_already_conserved() -> None:
    """A state already on both level sets is left untouched."""
    state = random_mps(SHAPES)
    reference = deepcopy(state)
    s2, sz = spin_mpos(4)
    targets = (expectation_value(state, s2), expectation_value(state, sz))

    assert not restore_invariants_joint(state, [s2, sz], targets, tol=TOL)
    for left, right in zip(state.tensors, reference.tensors, strict=True):
        assert np.array_equal(left, right)


def test_joint_solve_rejects_mismatched_lengths() -> None:
    """Observables and targets must be paired."""
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    with pytest.raises(ValueError, match="same length"):
        restore_invariants_joint(state, [s2, sz], [0.0], tol=TOL)


def test_joint_solve_with_no_observables_is_a_no_op() -> None:
    """An empty observable list displaces nothing."""
    state = random_mps(SHAPES)
    assert not restore_invariants_joint(state, [], [], tol=TOL)


# --- Routing and frames ---


def test_apply_conservation_routes_on_the_target_type() -> None:
    """An energy target takes the scalar solve and a two-observable target the joint one."""
    mpo = MPO.ising(4, 1.0, 0.5)
    s2, sz = spin_mpos(4)

    state = random_mps(SHAPES)
    conservation_target = EnergyTarget(expectation_value(state, mpo) * (1 + 1e-6), TOL)
    assert apply_conservation(state, mpo, conservation_target)
    assert expectation_value(state, mpo) == pytest.approx(conservation_target.value, abs=1e-12)

    state = random_mps(SHAPES)
    targets = ConservationTargets(
        names=("S2", "Sz"),
        mpos=(s2, sz),
        values=(expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3),
        tol=TOL,
    )
    assert apply_conservation(state, mpo, targets)
    assert expectation_value(state, s2) == pytest.approx(targets.values[0], abs=1e-10)
    assert expectation_value(state, sz) == pytest.approx(targets.values[1], abs=1e-10)


def test_sequential_routing_is_lopsided_where_joint_is_not() -> None:
    """``joint=False`` reproduces the last-wins behaviour, on the same targets."""
    s2, sz = spin_mpos(4)
    mpo = MPO.ising(4, 1.0, 0.5)
    state = random_mps(SHAPES)
    values = (expectation_value(state, s2) * (1 + 1e-3), expectation_value(state, sz) + 1e-3)
    targets = ConservationTargets(names=("S2", "Sz"), mpos=(s2, sz), values=values, tol=TOL, joint=False)

    assert apply_conservation(state, mpo, targets)
    assert expectation_value(state, sz) == pytest.approx(values[1], abs=1e-12)
    assert abs(expectation_value(state, s2) - values[0]) > 1e-9


def test_reflected_targets_reflect_every_observable() -> None:
    """BUG's second half-sweep runs reflected, so the observables must be too."""
    s2, sz = spin_mpos(4)
    targets = ConservationTargets(names=("S2", "Sz"), mpos=(s2, sz), values=(1.0, 2.0), tol=TOL)
    reflected = targets.reflected()

    assert reflected.values == targets.values
    assert reflected.names == targets.names
    for mine, theirs in zip(reflected.mpos, (s2.reflected(), sz.reflected()), strict=True):
        for left, right in zip(mine.tensors, theirs.tensors, strict=True):
            assert np.array_equal(left, right)


def test_capture_records_the_initial_expectation_values() -> None:
    """The recorded targets are the expectation values of the state that is evolved."""
    state = random_mps(SHAPES)
    s2, sz = spin_mpos(4)
    targets = record_conservation_targets(state, ["S2", "Sz"], [s2, sz], tol=TOL)

    assert targets.names == ("S2", "Sz")
    assert targets.values[0] == pytest.approx(expectation_value(state, s2))
    assert targets.values[1] == pytest.approx(expectation_value(state, sz))


def test_capture_rejects_mismatched_lengths() -> None:
    """Names and observables must be paired."""
    state = random_mps(SHAPES)
    s2, _ = spin_mpos(4)
    with pytest.raises(ValueError, match="same length"):
        record_conservation_targets(state, ["S2", "Sz"], [s2], tol=TOL)


# --- Integration with the BUG step ---


def test_uncompressed_run_conserves_both_observables() -> None:
    """A joint-corrected run holds both observables at their initial values."""
    state = random_mps([(2, 1, 2), (2, 2, 4), (2, 4, 2), (2, 2, 1)])
    mpo = MPO.ising(4, 1.0, 0.5)
    s2, sz = spin_mpos(4)
    targets = record_conservation_targets(state, ["S2", "Sz"], [s2, sz], tol=TOL)
    params = AnalogSimParams(
        elapsed_time=0.5, dt=0.05, max_bond_dim=2, svd_threshold=0.0, krylov_tol=1e-12, conserve_energy=True
    )

    for _ in range(10):
        bug(state, mpo, params, conservation_target=targets)

    assert expectation_value(state, s2) == pytest.approx(targets.values[0], abs=1e-9)
    assert expectation_value(state, sz) == pytest.approx(targets.values[1], abs=1e-9)


def untruncated_params() -> AnalogSimParams:
    """Parameters under which the compression discards nothing.

    Returns:
        Analog parameters with no bond cap and a zero discarded-weight threshold.
    """
    return AnalogSimParams(
        elapsed_time=0.5, dt=0.05, max_bond_dim=None, svd_threshold=0.0, krylov_tol=1e-14, conserve_energy=True
    )


def product_state(bits: str) -> MPS:
    """Build the computational-basis product state named by ``bits``.

    A basis state occupies exactly one U(1) charge sector, which is what separates the two
    magnetization tests below.

    Returns:
        The product state, canonical with the center at site 0.
    """
    up = np.array([1, 0], dtype=np.complex128)
    down = np.array([0, 1], dtype=np.complex128)
    tensors = [(down if bit == "1" else up).reshape(2, 1, 1) for bit in bits]
    state = MPS(len(bits), tensors=tensors)
    state.set_center(0)
    return state


def drift_without_truncation(mpo: MPO, observable: MPO, state: MPS | None = None, steps: int = 10) -> float:
    """Return the drift of ``observable`` over an uncorrected, untruncated run.

    Returns:
        The signed change in the normalized expectation value.
    """
    if state is None:
        state = random_mps([(2, 1, 2), (2, 2, 4), (2, 4, 2), (2, 2, 1)])
    start = expectation_value(state, observable)
    params = untruncated_params()
    for _ in range(steps):
        bug(state, mpo, params, conservation_target=None)
    return expectation_value(state, observable) - start


def test_energy_pin_is_inert_without_truncation() -> None:
    """With nothing discarded the energy is already at its target, so nothing is displaced.

    This is the property the manuscript's correction rests on: the uncompressed half-sweep
    preserves the energy, so the whole of the energy drift is charged to the compression and
    a corrected run with no compression is bit-identical to an uncorrected one.
    """
    mpo = MPO.ising(4, 1.0, 0.5)
    assert abs(drift_without_truncation(mpo, mpo)) < 1e-12

    corrected = random_mps([(2, 1, 2), (2, 2, 4), (2, 4, 2), (2, 2, 1)])
    targets = record_conservation_targets(corrected, ["H"], [mpo], tol=TOL)
    uncorrected = deepcopy(corrected)
    params = untruncated_params()
    for _ in range(5):
        bug(corrected, mpo, params, conservation_target=targets)
        bug(uncorrected, mpo, params, conservation_target=None)

    for left, right in zip(corrected.tensors, uncorrected.tensors, strict=True):
        assert np.array_equal(left, right)


def test_abelian_charge_is_preserved_from_a_charge_eigenstate() -> None:
    """From a definite charge sector the sweep preserves ``<S^z>`` structurally.

    ``S^z`` is an on-site additive charge, and a state that occupies one sector stays in it
    under a charge-conserving sweep. The drift lands far *below* round-off rather than at it,
    which is the signature of an exact structural conservation rather than a numerical
    accident, so a correction has nothing to repair and never fires.

    This is the condition that matters: it is a property of the state as well as the operator.
    See :func:`test_abelian_charge_drifts_from_a_superposition` for what happens without it.
    """
    heisenberg = heisenberg_mpo(4, delta=0.5)
    _, sz = spin_mpos(4)
    assert abs(drift_without_truncation(heisenberg, sz, product_state("0101"))) < 1e-20

    corrected = product_state("0101")
    targets = record_conservation_targets(corrected, ["Sz"], [sz], tol=TOL)
    uncorrected = deepcopy(corrected)
    params = untruncated_params()
    for _ in range(5):
        bug(corrected, heisenberg, params, conservation_target=targets)
        bug(uncorrected, heisenberg, params, conservation_target=None)

    for left, right in zip(corrected.tensors, uncorrected.tensors, strict=True):
        assert np.array_equal(left, right)


def test_abelian_charge_drifts_from_a_superposition() -> None:
    """Spanning several charge sectors, the sweep leaks weight between them.

    The conservation of the previous test is not a property of ``S^z`` alone. A state spread
    over several sectors has an ``<S^z>`` that is a weighted mean over them, and the
    integrator's time-discretization error moves that weight, so the mean drifts.
    """
    heisenberg = heisenberg_mpo(4, delta=0.5)
    _, sz = spin_mpos(4)
    assert abs(drift_without_truncation(heisenberg, sz)) > 1e-12


def test_total_spin_pin_is_not_inert_without_truncation() -> None:
    """The total spin is *not* preserved by the sweep, so its pin fires with nothing to fix.

    ``[S^2, H] = 0`` holds for the isotropic chain, but the sweep still moves ``<S^2>`` by many
    orders more than it moves the energy. A correction applied there is repairing the
    integrator's own time-discretization error rather than the compression, and it displaces
    the state on a step that discarded nothing. This is the asymmetry between ``S^2`` and both
    the energy and an abelian charge, and it is the reason the two tests above hold and this
    one cannot.
    """
    isotropic = heisenberg_mpo(4, delta=1.0)
    s2, _ = spin_mpos(4)
    drift = abs(drift_without_truncation(isotropic, s2))
    assert drift > 1e-12
    assert abs(drift_without_truncation(isotropic, isotropic)) < drift

    corrected = random_mps([(2, 1, 2), (2, 2, 4), (2, 4, 2), (2, 2, 1)])
    targets = record_conservation_targets(corrected, ["S2"], [s2], tol=TOL)
    uncorrected = deepcopy(corrected)
    params = untruncated_params()
    for _ in range(5):
        bug(corrected, isotropic, params, conservation_target=targets)
        bug(uncorrected, isotropic, params, conservation_target=None)

    displaced = any(
        not np.array_equal(left, right) for left, right in zip(corrected.tensors, uncorrected.tensors, strict=True)
    )
    assert displaced
    assert expectation_value(corrected, s2) == pytest.approx(targets.values[0], abs=1e-9)


# --- The configuration surface ---


def test_conserve_observables_defaults_to_empty() -> None:
    """The multi-observable feature is off by default and the joint mode on."""
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1)
    assert params.conserve_observables == {}
    assert params.conserve_joint is True


def test_capture_returns_none_with_nothing_conserved() -> None:
    """With no conservation configured the capture returns ``None``."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1)
    assert capture_conservation_target(state, MPO.ising(2, 1.0, 0.5), None, params) is None


def test_capture_keeps_the_energy_only_path_on_conservation_target() -> None:
    """``conserve_energy`` alone still produces an :class:`EnergyTarget`.

    The type is the guarantee: an :class:`EnergyTarget` takes exactly the code path that
    existed before the multi-observable feature, so the legacy behavior is bit-identical.
    """
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    mpo = MPO.ising(2, 1.0, 0.5)
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_energy=True)
    target = capture_conservation_target(state, mpo, None, params)
    assert isinstance(target, EnergyTarget)
    assert target.value == pytest.approx(expectation_value(state, mpo))


def test_capture_builds_joint_targets_with_energy_first() -> None:
    """With observables configured the energy leads the joint target set."""
    state = random_mps(SHAPES)
    mpo = MPO.ising(4, 1.0, 0.5)
    _, sz = spin_mpos(4)
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_energy=True, conserve_observables={"Sz": sz})
    target = capture_conservation_target(state, mpo, None, params)
    assert isinstance(target, ConservationTargets)
    assert target.names == ("H", "Sz")
    assert target.joint is True
    assert target.values[0] == pytest.approx(expectation_value(state, mpo))
    assert target.values[1] == pytest.approx(expectation_value(state, sz))


def test_capture_supports_observables_without_the_energy() -> None:
    """``conserve_observables`` works with ``conserve_energy`` off."""
    state = random_mps(SHAPES)
    _, sz = spin_mpos(4)
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_observables={"Sz": sz})
    target = capture_conservation_target(state, MPO.ising(4, 1.0, 0.5), None, params)
    assert isinstance(target, ConservationTargets)
    assert target.names == ("Sz",)


def test_capture_rejects_observables_under_noise() -> None:
    """A noise model refuses the multi-observable correction as it does the energy one."""
    state = random_mps([(2, 1, 2), (2, 2, 1)])
    _, sz = spin_mpos(4)
    params = AnalogSimParams(elapsed_time=0.1, dt=0.1, conserve_observables={"Sz": sz})
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="noise model"):
        capture_conservation_target(state, MPO.ising(2, 1.0, 0.5), noise, params)


# --- The zero-variance drop ---


def test_zero_variance_observable_is_dropped_from_the_joint_solve() -> None:
    """An eigenstate observable reduces the joint system to the remaining ones.

    From a state in a definite charge sector the magnetization gradient vanishes, so
    joint ``(H, S^z)`` must collapse to the scalar energy solve: the run is bit-identical
    to one that conserves the energy alone.
    """
    heisenberg = heisenberg_mpo(4, delta=0.5)
    _, sz = spin_mpos(4)
    params = AnalogSimParams(
        elapsed_time=0.5, dt=0.05, max_bond_dim=2, svd_threshold=0.0, krylov_tol=1e-14, conserve_energy=True
    )

    with_charge = product_state("0101")
    joint_targets = record_conservation_targets(with_charge, ["H", "Sz"], [heisenberg_mpo(4, delta=0.5), sz], tol=TOL)
    energy_only = product_state("0101")
    conservation_target = EnergyTarget(expectation_value(energy_only, heisenberg), TOL)

    for _ in range(10):
        bug(with_charge, heisenberg, params, conservation_target=joint_targets)
        bug(energy_only, heisenberg_mpo(4, delta=0.5), params, conservation_target=conservation_target)

    for left, right in zip(with_charge.tensors, energy_only.tensors, strict=True):
        assert np.array_equal(left, right)
