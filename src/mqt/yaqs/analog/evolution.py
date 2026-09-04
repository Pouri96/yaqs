# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared unitary evolution dispatch for analog TJM and ensemble paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.data_structures.simulation_parameters import EvolutionMode
from ..core.methods.bug import bug
from ..core.methods.conservation import apply_conservation
from ..core.methods.tdvp import tdvp

if TYPE_CHECKING:
    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams
    from ..core.methods.conservation import ConservationTargets, EnergyTarget


def apply_unitary_evolution(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams,
    *,
    normalize: bool = True,
    conservation_target: EnergyTarget | ConservationTargets | None = None,
) -> None:
    """Advance one unitary time step according to ``sim_params.evolution_mode``.

    When ``conservation_target`` is given, the conservation correction is applied once per step at
    the orthogonality center the compression leaves. BUG compresses twice per step, in the
    ordinary and the reflected frame, so it corrects inside the step and the target is
    passed down. TDVP ends every sweep with the center at site ``0`` and is corrected here.

    Args:
        state: MPS to evolve in place.
        hamiltonian: Time-independent Hermitian Hamiltonian as an MPO.
        sim_params: Analog simulation parameters (time step, bond limits, etc.).
        normalize: When ``evolution_mode`` is BUG, renormalize after compression.
            Ordinary physical states keep the default ``True``. Auxiliary correlator
            states (``B|ψ⟩``) should pass ``False`` so non-unitary probe amplitudes
            are preserved. TDVP ignores this flag.
        conservation_target: Conservation correction settings from
            :func:`~mqt.yaqs.analog.analog_tjm.capture_conservation_target`, or ``None``
            (default) to leave the step uncorrected.

    Raises:
        ValueError: If ``evolution_mode`` is not supported.
    """
    if sim_params.evolution_mode == EvolutionMode.TDVP:
        tdvp(state, hamiltonian, sim_params)
        if conservation_target is not None:
            state.assert_center(0, context="conservation correction after TDVP")
            apply_conservation(state, hamiltonian, conservation_target)
    elif sim_params.evolution_mode == EvolutionMode.BUG:
        bug(state, hamiltonian, sim_params, normalize=normalize, conservation_target=conservation_target)
    else:
        msg = f"Unsupported evolution_mode: {sim_params.evolution_mode!r}"
        raise ValueError(msg)
