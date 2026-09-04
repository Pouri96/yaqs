#!/usr/bin/env python3
"""Length-parametrized model, observable, and initial-state fixtures.

The manuscript runner ``l16_matched_optimized_2026-08-12/run_benchmark.py`` and the
energy-correction fixtures both close over a module-level ``LENGTH = 16``, so neither can
build a six-site system. This module rebuilds the same operators with the chain length as an
argument, in the same uncompressed finite-state-machine style and the same site-0-least-
significant-bit sparse convention, so that at ``length=16`` every construction here reproduces
the manuscript's operator exactly. ``check_fixtures.py`` asserts that.

Beyond the manuscript's TFIM and Haldane-Shastry it adds:

- ``xxx``, the isotropic Heisenberg chain. The energy-correction study's XXZ chain is fixed at
  ``DELTA = 0.5``, which carries only a U(1) symmetry; the isotropic point ``DELTA = 1`` is
  SU(2)-symmetric and is the model whose total spin is a genuine invariant.
- ``s2_mpo`` / ``sz_mpo``, the total spin and total magnetization as MPOs, for the correction
  to act on.
- ``sparse_s2`` / ``sparse_sz``, the same two observables assembled independently in the dense
  basis, for measurement. Correcting and measuring through one construction would hide an error
  in that construction, so the two paths are kept separate throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "COUPLING",
    "DELTA_XXZ",
    "FIELD",
    "INITIAL_CHI",
    "INITIAL_STATES",
    "MODELS",
    "NOISE_SCALE",
    "SEED",
    "TILT_ANGLE",
    "haldane_shastry_mpo",
    "hamiltonian_mpo",
    "initial_state",
    "is_su2_symmetric",
    "ising_mpo",
    "s2_mpo",
    "sparse_hamiltonian",
    "sparse_s2",
    "sparse_sx",
    "sparse_sy",
    "sparse_sz",
    "sx_mpo",
    "sy_mpo",
    "sz_mpo",
    "xxz_mpo",
]

#: Models the screen runs. ``tfim`` has no spin-rotation symmetry and ``xxz`` only a U(1) one,
#: so total spin is not conserved by either; they are the negative controls. ``xxx`` and ``hs``
#: are SU(2)-symmetric.
MODELS = ("tfim", "xxz", "xxx", "hs")

#: Initial states the screen runs, plus the tilted Neel state of the L=16 conservation table.
INITIAL_STATES = ("neel", "plus", "domain_wall", "random_product", "random_entangled", "tilted_neel")

#: Rotation angle of the ``tilted_neel`` state about the y axis.
TILT_ANGLE = np.pi / 4

COUPLING = 1.0
FIELD = 1.05
#: Anisotropy of the ``xxz`` arm, the manuscript's value: away from the isotropic point and
#: away from the free-fermion point, so the quench is neither of the two solvable cases.
DELTA_XXZ = 0.5
INITIAL_CHI = 4
NOISE_SCALE = 1e-10
SEED = 20260812

_ID = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2
_SPIN_OPS = (_SX, _SY, _SZ)


def is_su2_symmetric(model: str) -> bool:
    """Return whether ``model`` commutes with the total spin.

    Args:
        model: One of :data:`MODELS`.

    Returns:
        ``True`` for the isotropic Heisenberg and Haldane-Shastry chains, ``False`` for the
        transverse-field Ising chain and the anisotropic XXZ chain.

    Raises:
        ValueError: If ``model`` is not a supported name.
    """
    if model not in MODELS:
        msg = f"model must be one of {MODELS!r}, got {model!r}."
        raise ValueError(msg)
    return model in {"xxx", "hs"}


def _finalize(tensors: list[NDArray[np.complex128]]) -> MPO:
    """Wrap ``(left, physical, physical, right)`` blocks as an MPO.

    Args:
        tensors: One block per site, indexed ``(left, physical, physical, right)``.

    Returns:
        The operator as an :class:`MPO`, whose tensors are indexed
        ``(physical, physical, left, right)``.
    """
    mpo = MPO()
    mpo.custom([np.transpose(tensor, (1, 2, 0, 3)).copy() for tensor in tensors], transpose=False)
    return mpo


def _chain(bulk: NDArray[np.complex128], length: int) -> list[NDArray[np.complex128]]:
    """Open a translation-invariant bulk block into a finite chain.

    The first site keeps only the entering row of the finite-state machine and the last site
    only its exiting column, which is the manuscript's uncompressed convention.

    Args:
        bulk: The repeated block, indexed ``(left, physical, physical, right)``.
        length: Number of sites.

    Returns:
        The per-site blocks of the open chain.
    """
    left = bulk[0:1, :, :, :].copy()
    right = bulk[:, :, :, -1:].copy()
    return [left, *(bulk.copy() for _ in range(length - 2)), right]


def ising_mpo(length: int) -> MPO:
    """Build the bond-3 transverse-field Ising MPO.

    ``H = -J sum_i Z_i Z_i+1 - g sum_i X_i`` in Pauli operators, the manuscript's model.

    Args:
        length: Number of sites.

    Returns:
        The Hamiltonian as an :class:`MPO`.
    """
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    bulk = np.zeros((3, 2, 2, 3), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 1] = -COUPLING * z
    bulk[0, :, :, 2] = -FIELD * x
    bulk[1, :, :, 2] = z
    bulk[2, :, :, 2] = _ID
    return _finalize(_chain(bulk, length))


def xxz_mpo(length: int, delta: float) -> MPO:
    """Build the bond-5 XXZ MPO.

    ``H = sum_i (Sx_i Sx_i+1 + Sy_i Sy_i+1 + delta Sz_i Sz_i+1)`` in spin-1/2 operators.
    ``delta = 1`` is the isotropic Heisenberg point and is SU(2)-symmetric.

    Args:
        length: Number of sites.
        delta: Anisotropy of the ``Sz Sz`` coupling.

    Returns:
        The Hamiltonian as an :class:`MPO`.
    """
    bulk = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 1], bulk[0, :, :, 2], bulk[0, :, :, 3] = _SX, _SY, _SZ
    bulk[1, :, :, 4], bulk[2, :, :, 4], bulk[3, :, :, 4] = _SX, _SY, delta * _SZ
    bulk[4, :, :, 4] = _ID
    return _finalize(_chain(bulk, length))


def haldane_shastry_mpo(length: int) -> MPO:
    """Build the uncompressed finite-state-machine Haldane-Shastry MPO.

    ``H = sum_{i<j} J(i,j) S_i . S_j`` with the inverse-square chord coupling
    ``J(i,j) = J (pi/L)^2 / sin^2(pi (j-i) / L)``. Every pair couples, so the MPO bond grows
    along the chain; the construction is the manuscript's, with the length as an argument.

    Args:
        length: Number of sites.

    Returns:
        The Hamiltonian as an :class:`MPO`.
    """
    tensors: list[NDArray[np.complex128]] = []
    for site in range(1, length + 1):
        left_dim = 1 if site == 1 else 3 * (site - 1) + 2
        right_dim = 1 if site == length else 3 * site + 2
        tensor = np.zeros((left_dim, 2, 2, right_dim), dtype=np.complex128)
        if site == 1:
            tensor[0, :, :, 0] = _ID
            for axis, operator in enumerate(_SPIN_OPS, start=1):
                tensor[0, :, :, axis] = operator
        elif site == length:
            for start in range(1, length):
                coupling = COUPLING * (np.pi / length) ** 2 / np.sin(np.pi * (site - start) / length) ** 2
                for axis, operator in enumerate(_SPIN_OPS, start=1):
                    tensor[(start - 1) * 3 + axis, :, :, 0] += coupling * operator
            tensor[left_dim - 1, :, :, 0] = _ID
        else:
            tensor[0, :, :, 0] = _ID
            for axis, operator in enumerate(_SPIN_OPS, start=1):
                tensor[0, :, :, (site - 1) * 3 + axis] = operator
            for start in range(1, site):
                coupling = COUPLING * (np.pi / length) ** 2 / np.sin(np.pi * (site - start) / length) ** 2
                for axis, operator in enumerate(_SPIN_OPS, start=1):
                    virtual = (start - 1) * 3 + axis
                    tensor[virtual, :, :, virtual] = _ID
                    tensor[virtual, :, :, right_dim - 1] += coupling * operator
            tensor[left_dim - 1, :, :, right_dim - 1] = _ID
        tensors.append(tensor)
    return _finalize(tensors)


def s2_mpo(length: int) -> MPO:
    """Build the bond-5 total-spin MPO.

    ``S^2 = (sum_i S_i)^2 = (3L/4) I + 2 sum_{i<j} S_i . S_j`` for spin-1/2, where the constant
    collects the on-site ``S_i . S_i = 3/4``. Every pair couples at equal strength, which the
    finite-state machine expresses by carrying each spin channel through the identity: the
    ``S^a`` channel is opened at site ``i``, propagated unchanged, and closed at site ``j``.
    The constant rides on the direct entering-to-exiting transition, contributing ``3/4`` once
    per site.

    Args:
        length: Number of sites.

    Returns:
        The total spin as an :class:`MPO`, with bond dimension 5.
    """
    bulk = np.zeros((5, 2, 2, 5), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 4] = 0.75 * _ID
    for axis, operator in enumerate(_SPIN_OPS, start=1):
        bulk[0, :, :, axis] = operator
        bulk[axis, :, :, axis] = _ID
        bulk[axis, :, :, 4] = 2.0 * operator
    bulk[4, :, :, 4] = _ID
    return _finalize(_chain(bulk, length))


def sx_mpo(length: int) -> MPO:
    """Build the bond-2 total ``S^x`` MPO.

    ``S^x = sum_i S^x_i``. For an SU(2)-symmetric chain every total-spin component is
    conserved, and as a sum of on-site operators each is an admissible correction target on
    the same footing as ``S^z``.

    Args:
        length: Number of sites.

    Returns:
        The total ``S^x`` as an :class:`MPO`, with bond dimension 2.
    """
    bulk = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 1] = _SX
    bulk[1, :, :, 1] = _ID
    return _finalize(_chain(bulk, length))


def sy_mpo(length: int) -> MPO:
    """Build the bond-2 total ``S^y`` MPO.

    ``S^y = sum_i S^y_i``, admissible on the same footing as the other total-spin components.

    Args:
        length: Number of sites.

    Returns:
        The total ``S^y`` as an :class:`MPO`, with bond dimension 2.
    """
    bulk = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 1] = _SY
    bulk[1, :, :, 1] = _ID
    return _finalize(_chain(bulk, length))


def sz_mpo(length: int) -> MPO:
    """Build the bond-2 total-magnetization MPO.

    ``S^z = sum_i S^z_i``, the generator of the U(1) symmetry that both spin chains carry.

    Args:
        length: Number of sites.

    Returns:
        The total magnetization as an :class:`MPO`, with bond dimension 2.
    """
    bulk = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    bulk[0, :, :, 0] = _ID
    bulk[0, :, :, 1] = _SZ
    bulk[1, :, :, 1] = _ID
    return _finalize(_chain(bulk, length))


def hamiltonian_mpo(length: int, model: str) -> MPO:
    """Return the Hamiltonian MPO for ``model``.

    Args:
        length: Number of sites.
        model: One of :data:`MODELS`.

    Returns:
        The Hamiltonian as an :class:`MPO`.

    Raises:
        ValueError: If ``model`` is not a supported name.
    """
    if model == "tfim":
        return ising_mpo(length)
    if model == "xxz":
        return xxz_mpo(length, DELTA_XXZ)
    if model == "xxx":
        return xxz_mpo(length, 1.0)
    if model == "hs":
        return haldane_shastry_mpo(length)
    msg = f"model must be one of {MODELS!r}, got {model!r}."
    raise ValueError(msg)


def _pair_terms(
    length: int,
    pairs: list[tuple[int, int, float]],
) -> tuple[list[NDArray[np.int64]], list[NDArray[np.int64]], list[NDArray[np.float64]]]:
    """Assemble ``sum_{(i,j)} c_ij S_i . S_j`` in the site-0-least-significant-bit basis.

    ``Sz Sz`` is diagonal with eigenvalue ``(1-2b_i)(1-2b_j)/4``; ``Sx Sx + Sy Sy`` exchanges an
    antialigned pair with amplitude ``1/2`` and annihilates an aligned one.

    Args:
        length: Number of sites.
        pairs: Triples ``(i, j, c_ij)`` with ``i < j``.

    Returns:
        Row, column, and value arrays for a COO assembly.
    """
    dim = 1 << length
    basis = np.arange(dim, dtype=np.int64)
    diagonal = np.zeros(dim, dtype=np.float64)
    rows: list[NDArray[np.int64]] = [basis]
    cols: list[NDArray[np.int64]] = [basis]
    data: list[NDArray[np.float64]] = [diagonal]
    for left, right, coupling in pairs:
        left_bits = (basis >> left) & 1
        right_bits = (basis >> right) & 1
        diagonal += coupling * (1 - 2 * left_bits) * (1 - 2 * right_bits) / 4
        source = basis[left_bits != right_bits]
        rows.append(source ^ (1 << left) ^ (1 << right))
        cols.append(source)
        data.append(np.full(source.size, coupling / 2, dtype=np.float64))
    return rows, cols, data


def _coo(
    length: int,
    rows: list[NDArray[np.int64]],
    cols: list[NDArray[np.int64]],
    data: list[NDArray[np.float64]],
) -> sparse.csr_matrix:
    """Sum duplicate entries into a CSR matrix.

    Args:
        length: Number of sites.
        rows: Row-index arrays.
        cols: Column-index arrays.
        data: Value arrays.

    Returns:
        The assembled matrix in CSR format.
    """
    dim = 1 << length
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)
    )
    return matrix.tocsr()


def sparse_hamiltonian(length: int, model: str) -> sparse.csr_matrix:
    """Build the Hamiltonian in the dense basis, independently of the MPO.

    Args:
        length: Number of sites.
        model: One of :data:`MODELS`.

    Returns:
        The Hamiltonian as a sparse CSR matrix of dimension ``2 ** length``.

    Raises:
        ValueError: If ``model`` is not a supported name.
    """
    if model not in MODELS:
        msg = f"model must be one of {MODELS!r}, got {model!r}."
        raise ValueError(msg)
    dim = 1 << length
    basis = np.arange(dim, dtype=np.int64)

    if model == "tfim":
        diagonal = np.zeros(dim, dtype=np.float64)
        for site in range(length - 1):
            zi = 1 - 2 * ((basis >> site) & 1)
            zj = 1 - 2 * ((basis >> (site + 1)) & 1)
            diagonal -= COUPLING * zi * zj
        rows, cols, data = [basis], [basis], [diagonal]
        for site in range(length):
            rows.append(basis ^ (1 << site))
            cols.append(basis)
            data.append(np.full(dim, -FIELD, dtype=np.float64))
        return _coo(length, rows, cols, data)

    if model == "hs":
        pairs = [
            (left, right, COUPLING * (np.pi / length) ** 2 / np.sin(np.pi * (right - left) / length) ** 2)
            for left in range(length)
            for right in range(left + 1, length)
        ]
        return _coo(length, *_pair_terms(length, pairs))

    # The XXZ family is isotropic in the exchange part and anisotropic only in Sz Sz, so it is
    # the isotropic nearest-neighbour operator plus the residual (delta - 1) Sz Sz.
    delta = 1.0 if model == "xxx" else DELTA_XXZ
    pairs = [(site, site + 1, 1.0) for site in range(length - 1)]
    rows, cols, data = _pair_terms(length, pairs)
    if delta != 1.0:
        residual = np.zeros(dim, dtype=np.float64)
        for site in range(length - 1):
            zi = 1 - 2 * ((basis >> site) & 1)
            zj = 1 - 2 * ((basis >> (site + 1)) & 1)
            residual += (delta - 1.0) * zi * zj / 4
        rows.append(basis)
        cols.append(basis)
        data.append(residual)
    return _coo(length, rows, cols, data)


def sparse_s2(length: int) -> sparse.csr_matrix:
    """Build the total spin in the dense basis, independently of the MPO.

    Args:
        length: Number of sites.

    Returns:
        ``S^2`` as a sparse CSR matrix of dimension ``2 ** length``.
    """
    dim = 1 << length
    pairs = [(left, right, 2.0) for left in range(length) for right in range(left + 1, length)]
    rows, cols, data = _pair_terms(length, pairs)
    rows.append(np.arange(dim, dtype=np.int64))
    cols.append(np.arange(dim, dtype=np.int64))
    data.append(np.full(dim, 0.75 * length, dtype=np.float64))
    return _coo(length, rows, cols, data)


def sparse_sx(length: int) -> sparse.csr_matrix:
    """Build the total ``S^x`` in the dense basis, independently of the MPO.

    Args:
        length: Number of sites.

    Returns:
        ``S^x`` as a sparse CSR matrix of dimension ``2 ** length``.
    """
    dim = 1 << length
    basis = np.arange(dim, dtype=np.int64)
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []
    for site in range(length):
        rows.append(basis ^ (1 << site))
        cols.append(basis)
        data.append(np.full(dim, 0.5, dtype=np.float64))
    return _coo(length, rows, cols, data)


def sparse_sy(length: int) -> sparse.csr_matrix:
    """Build the total ``S^y`` in the dense basis, independently of the MPO.

    ``S^y_i`` flips bit ``i`` with amplitude ``+i/2`` from spin up and ``-i/2`` from spin
    down, in the site-0-least-significant-bit convention with bit 0 as spin up.

    Args:
        length: Number of sites.

    Returns:
        ``S^y`` as a sparse CSR matrix of dimension ``2 ** length``.
    """
    dim = 1 << length
    basis = np.arange(dim, dtype=np.int64)
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.complex128]] = []
    for site in range(length):
        bits = (basis >> site) & 1
        rows.append(basis ^ (1 << site))
        cols.append(basis)
        data.append((0.5j * (1 - 2 * bits)).astype(np.complex128))
    matrix = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)
    )
    return matrix.tocsr()


def sparse_sz(length: int) -> sparse.csr_matrix:
    """Build the total magnetization in the dense basis, independently of the MPO.

    Args:
        length: Number of sites.

    Returns:
        ``S^z`` as a sparse CSR matrix of dimension ``2 ** length``.
    """
    dim = 1 << length
    basis = np.arange(dim, dtype=np.int64)
    diagonal = np.zeros(dim, dtype=np.float64)
    for site in range(length):
        diagonal += (1 - 2 * ((basis >> site) & 1)) / 2
    return sparse.coo_matrix((diagonal, (basis, basis)), shape=(dim, dim)).tocsr()


def _local_vectors(length: int, name: str, rng: np.random.Generator) -> list[NDArray[np.complex128]]:
    """Return the single-site spinors of a product initial state.

    Args:
        length: Number of sites.
        name: One of the product entries of :data:`INITIAL_STATES`.
        rng: Source of randomness for ``random_product``.

    Returns:
        One normalized two-component spinor per site.
    """
    up = np.array([1, 0], dtype=np.complex128)
    down = np.array([0, 1], dtype=np.complex128)
    if name == "neel":
        return [up.copy() if site % 2 == 0 else down.copy() for site in range(length)]
    if name == "plus":
        return [np.array([1, 1], dtype=np.complex128) / np.sqrt(2) for _ in range(length)]
    # A sharp wall at the chain centre: the left half up, the right half down.
    if name == "domain_wall":
        return [up.copy() if site < length // 2 else down.copy() for site in range(length)]
    # The Neel state with every spin rotated by TILT_ANGLE about y. Unlike the Neel state it
    # spreads over several U(1) charge sectors, so the total magnetization has nonzero
    # variance and its conservation is a nontrivial property of a run.
    if name == "tilted_neel":
        half = TILT_ANGLE / 2.0
        rotated_up = np.array([np.cos(half), np.sin(half)], dtype=np.complex128)
        rotated_down = np.array([-np.sin(half), np.cos(half)], dtype=np.complex128)
        return [rotated_up.copy() if site % 2 == 0 else rotated_down.copy() for site in range(length)]
    vectors = []
    for _ in range(length):
        vector = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        vectors.append(np.asarray(vector / np.linalg.norm(vector), dtype=np.complex128))
    return vectors


def initial_state(length: int, name: str, *, chi: int = INITIAL_CHI, seed: int = SEED) -> MPS:
    """Build one initial MPS, padded and right-canonicalized.

    The four product states follow the manuscript's protocol: each bond is padded to ``chi``
    with seeded ``NOISE_SCALE`` entries, the chain is right-canonicalized by QR, and the centre
    is left at site ``0``. Padding is what lets a rank-adaptive sweep grow the bond at all, and
    ``chi`` is exposed because the start bond is one of the screen's controls.

    ``random_entangled`` is built directly at bond ``chi`` instead, so it does not sit on the
    product manifold at ``t = 0``.

    Args:
        length: Number of sites.
        name: One of :data:`INITIAL_STATES`.
        chi: Bond dimension to pad to. ``1`` leaves a product state unpadded.
        seed: Seed of the padding and random-state generator.

    Returns:
        The initial state, right-canonical with the centre at site ``0``.

    Raises:
        ValueError: If ``name`` is not a supported initial state.
    """
    if name not in INITIAL_STATES:
        msg = f"name must be one of {INITIAL_STATES!r}, got {name!r}."
        raise ValueError(msg)
    rng = np.random.default_rng(seed + INITIAL_STATES.index(name))

    if name == "random_entangled":
        bonds = [1, *[min(chi, 2 ** min(site + 1, length - site - 1)) for site in range(length - 1)], 1]
        tensors = []
        for site in range(length):
            shape = (bonds[site], 2, bonds[site + 1])
            block = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
            tensors.append(np.asarray(block, dtype=np.complex128))
    else:
        tensors = [vector.reshape(1, 2, 1).copy() for vector in _local_vectors(length, name, rng)]
        for bond in range(length - 1):
            left, right = tensors[bond], tensors[bond + 1]
            old = left.shape[2]
            if old >= chi:
                continue
            padded_left = np.zeros((left.shape[0], left.shape[1], chi), dtype=np.complex128)
            padded_left[:, :, :old] = left
            shape = padded_left[:, :, old:].shape
            padded_left[:, :, old:] = NOISE_SCALE * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            padded_right = np.zeros((chi, right.shape[1], right.shape[2]), dtype=np.complex128)
            padded_right[:old, :, :] = right
            shape = padded_right[old:, :, :].shape
            padded_right[old:, :, :] = NOISE_SCALE * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            tensors[bond], tensors[bond + 1] = padded_left, padded_right

    for site in range(length - 1, 0, -1):
        tensor = tensors[site]
        left, physical, right = tensor.shape
        qh, rh = np.linalg.qr(tensor.reshape(left, physical * right).conj().T, mode="reduced")
        transfer = rh.conj().T
        tensors[site] = qh.conj().T.reshape(qh.shape[1], physical, right)
        previous = tensors[site - 1]
        tensors[site - 1] = (previous.reshape(-1, left) @ transfer).reshape(
            previous.shape[0], previous.shape[1], transfer.shape[1]
        )
    tensors[0] /= np.linalg.norm(tensors[0])

    state = MPS(length, tensors=[np.transpose(tensor, (1, 0, 2)).copy() for tensor in tensors])
    state.set_center(0)
    return state
