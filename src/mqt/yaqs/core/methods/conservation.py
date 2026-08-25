# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Restores a conserved expectation value after an SVD compression sweep.

An uncompressed BUG half-sweep preserves ``<psi|H|psi>``. The compression map applied
after each half-sweep discards weight from the state and with it a portion of that
quantity, so the compressed composition conserves neither the norm nor the energy. This
module appends a correction to the compression map that restores the energy exactly and
changes no bond dimension.

After a compression sweep the MPS is canonical with the orthogonality center ``C`` at the
endpoint the sweep leaves. The global expectation value then equals a Rayleigh quotient of
the effective operator ``H_eff = J^dagger H J`` on that single tensor,

    <psi|H|psi> / <psi|psi>  =  <C, H_eff C>_F / <C, C>_F,

where ``H_eff`` acts through the same MPO environment contractions the local update uses
and is never formed explicitly. Writing ``E`` for the post-compression value and
``g = (H_eff - E) C``, so that ``<C, g>_F = 0``, the displacement ``C -> C + mu g`` with
``mu`` the real root of smallest modulus of

    a mu^2 + b mu + c = 0,
    a = <g, H_eff g>_F - target ||g||_F^2,
    b = 2 ||g||_F^2,
    c = <C, C>_F (E - target),

restores the quotient to ``target``. The linear coefficient is exact because
``<g, H_eff C>_F = ||g||_F^2`` follows from ``<C, g>_F = 0``, so it costs no additional
contraction and is positive whenever ``g != 0``. The root is evaluated in factored form to
avoid cancellation for small ``a``. Two matrix-free applications of ``H_eff``, one to ``C``
and one to ``g``, complete the solve.

Placement is part of the construction, not a matter of style. The correction acts at the
orthogonality center the compression sweep leaves, after the truncation and before the
center is renormalized. The Rayleigh quotient is scale invariant, so the subsequent rescale
leaves the restored value unchanged. Transporting the center across a compressed bond would
rotate the block bases the sweep produced and invalidate the orthogonality on which the
correction's error identity rests.

The correction is structure preservation. It restores the invariant exactly; it does not
reduce the compression error, and it cannot: the discarded component is orthogonal to every
state the retained bases represent, while the displacement ``mu J g`` lies among them, so
the distance to the uncompressed state grows by ``mu^2 ||g||_F^2`` for every ``mu != 0``.

Two limits fix the scope. The construction presumes a time-independent Hermitian
Hamiltonian, and only invariants of the projected flow are reachable: ``[O, H] = 0`` does
not imply ``[O_eff, H_eff] = 0``. Under a noise model ``<H>`` is not an invariant of the
trajectory at all, and :func:`validate_energy_conservation` rejects that combination.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .tdvp.primitives import project_site, update_left_environment, update_right_environment

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams

__all__ = [
    "energy_expectation",
    "restore_energy_at_center",
    "solve_min_modulus_root",
    "validate_energy_conservation",
]

# Below this squared Frobenius norm the center tensor or the gradient carries no usable
# direction and the correction is skipped rather than divided by.
_ZERO_NORM_SQUARED = 1e-300
_ZERO_GRADIENT_SQUARED = 1e-28


def solve_min_modulus_root(a: float, b: float, c: float) -> float | None:
    """Return the real root of ``a x^2 + b x + c`` with the smallest modulus.

    Evaluated as ``c / q`` with ``q = -(b + sign(b) sqrt(b^2 - 4ac)) / 2``. The schoolbook
    expression cancels catastrophically on this branch as ``a -> 0``, which is the ordinary
    regime here because ``a`` carries only the curvature of the quotient while ``b`` is
    twice a squared norm. The factored form also needs no separate ``a == 0`` case.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant coefficient.

    Returns:
        The real root of smallest modulus, or ``None`` if the discriminant is negative or
        the factored denominator underflows.
    """
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    # ``b`` is twice a squared norm wherever the correction runs, so it is positive; the
    # branch only matters for the direct calls in the tests. Zero takes the positive sign.
    sign_b = -1.0 if b < 0.0 else 1.0
    q = -0.5 * (b + sign_b * math.sqrt(discriminant))
    if abs(q) < _ZERO_NORM_SQUARED:
        return None
    return c / q


def _boundary_block(bond_dim: int, mpo_bond_dim: int) -> NDArray[np.complex128]:
    """Build the identity environment block closing one end of the chain.

    Args:
        bond_dim: Virtual bond dimension of the state at that end.
        mpo_bond_dim: Virtual bond dimension of the MPO at that end.

    Returns:
        Block of shape ``(bond_dim, mpo_bond_dim, bond_dim)`` that is the identity on the
        state legs for every MPO index.
    """
    identity = np.eye(bond_dim, dtype=np.complex128)
    return np.asarray(np.tile(identity[:, np.newaxis, :], (1, mpo_bond_dim, 1)), dtype=np.complex128)


def _left_environment(state: MPS, mpo: MPO, site: int) -> NDArray[np.complex128]:
    """Contract the chain left of ``site`` into an MPO environment block.

    The contraction uses the state tensors as both ket and bra, so the result is exact for
    any gauge; no isometry is assumed.

    Args:
        state: The MPS.
        mpo: The MPO in the same frame as ``state``.
        site: Site whose left environment is built.

    Returns:
        The left environment block at ``site``.
    """
    block = _boundary_block(state.tensors[0].shape[1], mpo.tensors[0].shape[2])
    for left_site in range(site):
        block = update_left_environment(
            state.tensors[left_site], state.tensors[left_site], mpo.tensors[left_site], block
        )
    return block


def _right_environment(state: MPS, mpo: MPO, site: int) -> NDArray[np.complex128]:
    """Contract the chain right of ``site`` into an MPO environment block.

    The contraction uses the state tensors as both ket and bra, so the result is exact for
    any gauge; no isometry is assumed.

    Args:
        state: The MPS.
        mpo: The MPO in the same frame as ``state``.
        site: Site whose right environment is built.

    Returns:
        The right environment block at ``site``.
    """
    block = _boundary_block(state.tensors[-1].shape[2], mpo.tensors[-1].shape[3])
    for right_site in range(state.length - 1, site, -1):
        block = update_right_environment(
            state.tensors[right_site], state.tensors[right_site], mpo.tensors[right_site], block
        )
    return block


def energy_expectation(state: MPS, mpo: MPO) -> float:
    """Return the normalized expectation value ``<psi|H|psi> / <psi|psi>``.

    Computed from MPO environment contractions, so it holds for any gauge and requires no
    canonical form. Use it once, before the step loop, to record the value the correction
    restores; it must be taken from the state that is actually evolved, after any padding
    of the initial MPS.

    Args:
        state: The MPS.
        mpo: Hermitian Hamiltonian as an MPO in the same frame as ``state``.

    Returns:
        The normalized expectation value.

    Raises:
        ValueError: If the site counts differ or the state has vanishing norm.
    """
    if mpo.length != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    tensor = state.tensors[0]
    projected = project_site(
        _left_environment(state, mpo, 0),
        _right_environment(state, mpo, 0),
        mpo.tensors[0],
        tensor,
    )
    norm_squared = float(state.scalar_product(state).real)
    if not np.isfinite(norm_squared) or norm_squared < _ZERO_NORM_SQUARED:
        msg = f"Cannot take an expectation value of a state with squared norm {norm_squared!r}."
        raise ValueError(msg)
    return float(np.vdot(tensor, projected).real) / norm_squared


def restore_energy_at_center(state: MPS, mpo: MPO, target: float, *, tol: float = 1e-12) -> bool:
    """Restore ``<H> = target`` by a rank-preserving displacement of the center tensor.

    The MPS must be canonical with a tracked orthogonality center, which is the state a
    compression sweep leaves. The center tensor is replaced in place by ``C + mu g``; its
    shape and every other tensor are unchanged, so all bond dimensions are preserved.

    Args:
        state: The MPS. The tensor at the tracked center is modified in place.
        mpo: Hermitian Hamiltonian as an MPO in the same frame as ``state``.
        target: Value to restore the normalized expectation value to.
        tol: Relative skip threshold. The correction is omitted while
            ``|E - target| < tol * max(1, |target|)``, which leaves a step that discarded
            no weight bit-identical. The guard is relative because an absolute threshold
            sits below the round-off of ``<H>`` itself once ``|target|`` is large.

    Returns:
        Whether the center tensor was displaced. ``False`` covers the skip cases: the
        invariant is already at ``target`` within ``tol``, the center is an eigenvector of
        the effective operator, or the level set does not meet the line ``{C + mu g}``.

    Raises:
        ValueError: If the site counts differ or the gauge is unknown.
    """
    if mpo.length != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    center = state.orthogonality_center
    if center is None:
        msg = "restore_energy_at_center: MPS gauge unknown (orthogonality_center is None)."
        raise ValueError(msg)

    center_tensor = state.tensors[center]
    norm_squared = float(np.vdot(center_tensor, center_tensor).real)
    if norm_squared < _ZERO_NORM_SQUARED:
        return False

    left_env = _left_environment(state, mpo, center)
    right_env = _right_environment(state, mpo, center)
    local_op = mpo.tensors[center]

    projected = project_site(left_env, right_env, local_op, center_tensor)
    quotient = float(np.vdot(center_tensor, projected).real) / norm_squared
    if abs(quotient - target) < tol * max(1.0, abs(target)):
        return False

    gradient = projected - quotient * center_tensor
    gradient_squared = float(np.vdot(gradient, gradient).real)
    if gradient_squared < _ZERO_GRADIENT_SQUARED:
        return False

    curvature = float(np.vdot(gradient, project_site(left_env, right_env, local_op, gradient)).real)
    multiplier = solve_min_modulus_root(
        curvature - target * gradient_squared,
        2.0 * gradient_squared,
        norm_squared * (quotient - target),
    )
    if multiplier is None:
        return False

    state.tensors[center] = np.asarray(center_tensor + multiplier * gradient, dtype=np.complex128)
    return True


def validate_energy_conservation(sim_params: AnalogSimParams | DigitalSimParams, *, has_noise: bool) -> None:
    """Reject an energy-conservation setting whose premise does not hold.

    Called once before a run starts, so a misconfigured simulation fails immediately rather
    than pinning the wrong value for its whole length. The correction restores ``<H>`` to
    its initial value, which presumes that ``<H>`` is an invariant of the evolution. Under a
    noise model it is not: dissipation and jumps move the energy of the physical trajectory.

    Args:
        sim_params: Simulation parameters carrying ``conserve_energy``.
        has_noise: Whether the run applies a noise model.

    Raises:
        ValueError: If ``conserve_energy`` is set on a run with a noise model.
    """
    if sim_params.conserve_energy and has_noise:
        msg = (
            "conserve_energy=True is invalid together with a noise model. The energy is not "
            "an invariant of a noisy trajectory, so restoring it to its initial value would "
            "drive the state toward a quantity the evolution does not preserve. Use "
            "conserve_energy=False for open-system runs."
        )
        raise ValueError(msg)
