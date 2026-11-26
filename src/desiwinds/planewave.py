"""
Plane-wave windows: the forward-model window as a deterministic integral over wavevectors.

The expected spectrum of the Gaussian mocks of :py:func:`desiwinds.forward.mock_survey_catalog` is, to second order in the mock density, the integral over wavevectors of the spectrum produced by single plane waves. Evaluated with a quadrature, it gives the window at any k, including below the fundamental mode of any mesh, with no seed noise.

Per wavevector node, :py:func:`get_planewave_responses` returns the responses R_00, R_02, R_22 of the estimator to the two tangents u_0 = exp(i k.x) and u_2 = L_2(k.n) exp(i k.x) (n the local line of sight), summed over +k and -k; the window columns of the theory multipoles, of spikes or of basis vectors, are weighted sums of them (:py:func:`get_planewave_window`).
"""

import dataclasses
import itertools
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxpower import BinMesh2SpectrumPoles, FKPField, MeshAttrs, ParticleField, compute_mesh2_spectrum, get_mesh_attrs
from jaxpower.mesh import get_sharding_mesh

from .forward import (
    AMR_args,
    NAM_args,
    RIC_args,
    _apply_effects,
    _check_effects_per_field,
    _fill_with_constant,
    _split_weights,
    _update_fkp,
    apply_AMR,
    apply_NAM,
    apply_RIC,
    bincount_sorted_vmapped,
    prepare_AMR,
    prepare_NAM,
    prepare_RIC,
)
from .utils import bincount, local_concatenate

PAINT_KWARGS = {"resampler": "tsc", "interlacing": 3, "compensate": True, "out": "real"}


def get_planewave_nodes(k_edges: np.ndarray, n_k: int = 4, n_directions: int = 64, log: bool = True) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Get quadrature nodes on a hemisphere of wavevectors and their weights for :math:`\int d^3k / (2\pi)^3`.

    Gauss-Legendre in :math:`\ln k` (``log=True``) or in :math:`k` on each segment ``[k_edges[i], k_edges[i+1]]``, times Fibonacci directions on the upper hemisphere (:math:`\hat k_z \geq 0`).
    The weights integrate over the half space; the responses of :py:func:`get_planewave_responses` are summed over :math:`\pm \mathbf{k}`, so the full integral is ``weights @ responses``.

    Parameters
    ----------
    k_edges : np.ndarray
        Edges of the wavenumber segments, shape ``(n_segments + 1,)``. One Gauss-Legendre quadrature is used per segment.
    n_k : int, optional
        Number of Gauss-Legendre nodes per segment. Default is 4.
    n_directions : int, optional
        Number of Fibonacci directions on the hemisphere. Default is 64.
    log : bool, optional
        Whether to integrate in :math:`\ln k` rather than :math:`k`. Requires ``k_edges > 0``. Default is True.

    Returns
    -------
    kvec : np.ndarray
        Wavevector nodes, shape ``(n_nodes, 3)``.
    weights : np.ndarray
        Quadrature weights of the nodes, shape ``(n_nodes,)``.

    Raises
    ------
    ValueError
        If ``log=True`` and an edge is not positive.
    """
    k_edges = np.asarray(k_edges, dtype=float)
    x, w = np.polynomial.legendre.leggauss(n_k)
    knorms, kweights = [], []
    for low, high in itertools.pairwise(k_edges):
        if log:
            if low <= 0:
                raise ValueError("log quadrature needs k_edges > 0")
            lnk = np.log(low) + (np.log(high) - np.log(low)) * (x + 1) / 2
            k = np.exp(lnk)
            knorms.append(k)
            kweights.append(w * (np.log(high) - np.log(low)) / 2 * k**3)  # d^3k = k^3 dlnk dOmega
        else:
            k = low + (high - low) * (x + 1) / 2
            knorms.append(k)
            kweights.append(w * (high - low) / 2 * k**2)
    knorms, kweights = np.concatenate(knorms), np.concatenate(kweights)
    # Fibonacci directions on the upper hemisphere, equal solid angle 2 pi / n
    i = np.arange(n_directions) + 0.5
    cos_theta = i / n_directions
    phi = np.pi * (1 + 5**0.5) * i
    sin_theta = np.sqrt(1 - cos_theta**2)
    directions = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=-1)
    kvec = (knorms[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    weights = (kweights[:, None] * np.full(n_directions, 2 * np.pi / n_directions)[None, :]).reshape(-1) / (2 * np.pi) ** 3
    return kvec, weights


def get_mesh_mode_nodes(mattrs: MeshAttrs, k_edges: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Get the modes of the mesh ``mattrs`` with :math:`k_a \leq |\mathbf{k}| < k_b`, on the half space, with weights :math:`1/V`.

    With :py:func:`get_planewave_responses` and ``readout="mesh"``, ``weights @ responses`` is the expectation of the forward-model spike window column of the bin (``unitary_amplitude=True``): the validation of the plane-wave window.

    Parameters
    ----------
    mattrs : MeshAttrs
        Mesh whose modes are returned.
    k_edges : tuple[float, float]
        Lower and upper edges :math:`(k_a, k_b)` of the bin.

    Returns
    -------
    kvec : np.ndarray
        Wavevectors of the modes, shape ``(n_modes, 3)``.
    weights : np.ndarray
        Weight :math:`1/V` of each mode (:math:`V` the volume of the box), shape ``(n_modes,)``.

    Notes
    -----
    Modes with :math:`\mathbf{k} \equiv -\mathbf{k}` (zero or Nyquist components) must not be in the bin.
    """
    kvec = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(int(n), d=float(c)) for n, c in zip(mattrs.meshsize, mattrs.cellsize, strict=True)], indexing="ij")
    kvec = np.stack([kk.ravel() for kk in kvec], axis=-1)
    knorm = np.linalg.norm(kvec, axis=-1)
    mask = (knorm >= k_edges[0]) & (knorm < k_edges[1])
    # half space: first non-zero component positive
    first = np.where(kvec[:, 0] != 0, kvec[:, 0], np.where(kvec[:, 1] != 0, kvec[:, 1], kvec[:, 2]))
    mask &= first > 0
    return kvec[mask], np.full(mask.sum(), 1.0 / float(np.prod(np.asarray(mattrs.boxsize))))


def halves_from_split(
    fkp_field: FKPField, ric_args: RIC_args | None = None, amr_args: AMR_args | None = None
) -> tuple[tuple[FKPField, FKPField], tuple[RIC_args, RIC_args] | None, tuple[AMR_args, AMR_args] | None]:
    r"""
    Get the two disjoint halves of an FM split, and their effect arguments, without gathering. A shortcut when you already have an FM split on hand.

    The FM splits the randoms at random into "data" and "randoms": half A is its data, half B its randoms (weights are put on the data half).
    Here, each half becomes an FKP field whose data and randoms are the same particles, and the two halves are crossed as the two legs of one pair.
    Their RIC / AMR arguments are those prepared for the FM split, with the randoms entries of A pointing to its data entries, and conversely for B, so nothing is prepared again.
    Works with sharded, multi-process particles. Without an FM split to reuse, use :py:func:`prepare_halves_effects`.

    Parameters
    ----------
    fkp_field : FKPField
        FKP field of the FM split: its data and randoms become the two halves.
    ric_args : RIC_args | None, optional
        RIC arguments prepared for the FM split with :py:func:`desiwinds.forward.prepare_RIC` (a single set, not one per tracer). Default is ``None``.
    amr_args : AMR_args | None, optional
        AMR arguments prepared for the FM split with :py:func:`desiwinds.forward.prepare_AMR`, which needs ``keep_data_templates_normalized=True`` (a single set, not one per tracer). Default is ``None``.

    Returns
    -------
    halves : tuple[FKPField, FKPField]
        To be passed as one pair to :py:func:`get_planewave_responses`.
    ric_args : tuple[RIC_args, RIC_args] | None
        RIC arguments of each half, ``None`` if not given, with ``data_regions``, ``randoms_regions`` for the renormalization.
    amr_args : tuple[AMR_args, AMR_args] | None
        AMR arguments of each half, ``None`` if not given.
    """
    data, randoms = fkp_field.data, fkp_field.randoms
    randoms = randoms.clone(weights=randoms.weights * (data.weights.sum() / randoms.weights.sum()))
    halves = (FKPField(data, data, attrs=fkp_field.attrs), FKPField(randoms, randoms, attrs=fkp_field.attrs))

    def as_half(args, side):
        """Get the arguments of the half made of the FM's ``side`` particles: both data_* and randoms_* entries from ``side``_*."""
        other = "randoms" if side == "data" else "data"
        changes = {}
        for field in dataclasses.fields(args):
            name = field.name
            if name == "data_to_remove":  # data and randoms coincide: no data without randoms
                reference = next(getattr(args, f"{side}_{suffix}") for suffix in ("distances_digitized", "pixels") if hasattr(args, f"{side}_{suffix}"))
                changes[name] = jnp.zeros(reference.shape, dtype=bool)
            elif name.startswith(other + "_") and getattr(args, side + name[len(other) :]) is not None:
                changes[name] = getattr(args, side + name[len(other) :])
        return dataclasses.replace(args, **changes)

    def both_halves(args):
        return None if args is None else (as_half(args, "data"), as_half(args, "randoms"))

    return halves, both_halves(ric_args), both_halves(amr_args)


def prepare_halves_effects(
    halves: tuple[FKPField, FKPField] | list[tuple[FKPField, FKPField]],
    ric_kwargs: dict | None = None,
    amr_kwargs: dict | None = None,
    nam_kwargs: dict | None = None,
) -> tuple[tuple[RIC_args, RIC_args] | None, tuple[AMR_args, AMR_args] | None, tuple[NAM_args, NAM_args] | None]:
    """
    Prepare the effect arguments of two disjoint halves, each used as both data and randoms, to cross them in :py:func:`get_planewave_responses`.

    Each half is prepared on its own, with its particles as both data and randoms: no FM split is needed. The halves themselves come from the caller, for example the particles of a catalog split at random into two halves with the same machinery as the FM's data / randoms split.

    Parameters
    ----------
    halves : tuple[FKPField, FKPField] | list[tuple[FKPField, FKPField]]
        The two halves, each an FKP field whose data and randoms are the same particles. For several caps that share their effects, one pair per cap (the effects of a half are then prepared jointly over the caps).
    ric_kwargs : dict | None, optional
        Keyword arguments of :py:func:`desiwinds.forward.prepare_RIC` (all but ``data`` and ``randoms``): ``regions``, ``n_bins``, ``apply_to``. Default is ``None`` (no RIC).
    amr_kwargs : dict | None, optional
        Keyword arguments of :py:func:`desiwinds.forward.prepare_AMR` (all but ``data`` and ``randoms``). ``keep_data_templates_normalized`` is set to True, as the data are used as randoms. The particles need the extra columns ``"Z"`` and ``"template_values"``. Default is ``None`` (no AMR).
    nam_kwargs : dict | None, optional
        Keyword arguments of :py:func:`desiwinds.forward.prepare_NAM` (all but ``data`` and ``randoms``). The particles need the extra column ``"Z"``. Default is ``None`` (no NAM).

    Returns
    -------
    ric_args : tuple[RIC_args, RIC_args] | None
        RIC arguments of each half, ``None`` if ``ric_kwargs`` is ``None``.
    amr_args : tuple[AMR_args, AMR_args] | None
        AMR arguments of each half, ``None`` if ``amr_kwargs`` is ``None``.
    nam_args : tuple[NAM_args, NAM_args] | None
        NAM arguments of each half, ``None`` if ``nam_kwargs`` is ``None``.

    Notes
    -----
    The binning of the effects (AMR template percentiles and bins) is computed on each half, whereas :py:func:`halves_from_split` reuses the one computed on the FM's randoms: the two routes agree statistically, not exactly (for RIC, whose bins only depend on the box, exactly).
    """
    pairs = [halves] if isinstance(halves[0], FKPField) else list(halves)

    def prepare(prepare_effect, kwargs, **forced):
        if kwargs is None:
            return None
        return tuple(
            prepare_effect(data=[pair[side].data for pair in pairs], randoms=[pair[side].randoms for pair in pairs], **(kwargs | forced)) for side in (0, 1)
        )

    return prepare(prepare_RIC, ric_kwargs), prepare(prepare_AMR, amr_kwargs, keep_data_templates_normalized=True), prepare(prepare_NAM, nam_kwargs)


def _amr_split(exploded: jax.Array, n_regions: int, axis: int = -1) -> jax.Array:
    """
    Split the exploded AMR bincounts per region, without the edge bins, as :py:func:`desiwinds.forward.apply_AMR` does.

    Parameters
    ----------
    exploded : jax.Array
        Bincounts, with the bins of all the regions concatenated along ``axis``.
    n_regions : int
        Number of regions.
    axis : int, optional
        Axis of the concatenated bins: the last (default, -1) or the second to last.

    Returns
    -------
    jax.Array
        Bincounts per region, with regions along a new leading axis and the two edge bins of each region removed.
    """
    binned = jnp.stack(jnp.split(exploded, n_regions, axis=axis))
    return binned[..., 1:-1] if axis in (-1, exploded.ndim - 1) else binned[..., 1:-1, :]


def _amr_coefficients(
    data_exploded: jax.Array, randoms_exploded: jax.Array, randoms_templates_exploded: jax.Array, data_sums: jax.Array, randoms_sums: jax.Array
) -> jax.Array:
    """
    Compute the AMR regression coefficients from the binned sums: the algebra of :py:func:`desiwinds.forward.apply_AMR`.

    Parameters
    ----------
    data_exploded : jax.Array
        Binned data weights, per template, with the bins of all regions concatenated, shape ``(n_sys, (n_bins + 2) * n_regions)``.
    randoms_exploded : jax.Array
        Binned randoms weights, same layout as ``data_exploded``.
    randoms_templates_exploded : jax.Array
        Binned randoms weights times the normalized templates, shape ``(n_sys, (n_bins + 2) * n_regions, n_sys + 1)``.
    data_sums : jax.Array
        Sum of the data weights per region, shape ``(n_regions,)``.
    randoms_sums : jax.Array
        Sum of the randoms weights per region, shape ``(n_regions,)``.

    Returns
    -------
    jax.Array
        Regression coefficients, shape ``(n_regions, n_sys + 1)``.
    """
    n_regions = data_sums.shape[0]
    data_binned = _amr_split(data_exploded, n_regions)
    randoms_binned = _amr_split(randoms_exploded, n_regions)
    randoms_templates_binned = _amr_split(randoms_templates_exploded, n_regions, axis=1)
    normalization = randoms_sums / data_sums
    nonzero = randoms_binned != 0
    # safe square root: no nan in the derivatives of empty bins
    prefactor = jnp.where(nonzero, jnp.sqrt(normalization[:, None, None] / jnp.where(nonzero, randoms_binned, 1.0)), 0.0)
    design = prefactor[..., None] * randoms_templates_binned / normalization[..., None, None, None]
    y = prefactor * (data_binned - randoms_binned / normalization[..., None, None])
    design_t_y = jnp.einsum("rijp, rij -> rp ", design, y)
    design_t_design = jnp.einsum("rijq, rijp -> rpq", design, design)
    return jnp.linalg.solve(design_t_design, design_t_y[..., None]).squeeze(-1)


def _ric_randoms_bins(ric: RIC_args) -> tuple[jax.Array, int]:
    """
    Get the index of the RIC (region, distance) bin of each random, the last index for randoms in no region.

    Parameters
    ----------
    ric : RIC_args
        RIC arguments (precomputed with :py:func:`desiwinds.forward.prepare_RIC`).

    Returns
    -------
    index : jax.Array
        Bin index of each random, shape ``(n_r,)``.
    n_bins : int
        Number of bins, including the one for randoms in no region.
    """
    n_regions = ric.randoms_regions.shape[0]
    index = jnp.full(ric.randoms_distances_digitized.shape, n_regions * ric.n_bins, dtype=jnp.int32)
    for iregion in range(n_regions):
        index = jnp.where(ric.randoms_regions[iregion], iregion * ric.n_bins + ric.randoms_distances_digitized.astype(jnp.int32) - 1, index)
    return index, n_regions * ric.n_bins + 1


@jax.custom_jvp
def _amr_weights(data_weights: jax.Array, randoms_weights: jax.Array, amr: AMR_args, ric: RIC_args | None) -> jax.Array:
    """
    Compute the AMR weights as :py:func:`desiwinds.forward.apply_AMR` does, with a cheap exact tangent.

    The tangent only needs the tangents of the binned sums the regression is built on.
    The data ones are bincounts of the data tangent; the randoms ones would cost (n_sys + 1)^2 bincounts of all the randoms per tangent, but before AMR the randoms weights only change through the RIC, by one factor per RIC (region, distance) bin: their tangent is randoms_weights x G[RIC bin], and the randoms sums are contractions of G with cross-tables of the RIC and AMR bins, computed once with the primal.

    Parameters
    ----------
    data_weights : jax.Array
        Data weights, shape ``(n_d,)``.
    randoms_weights : jax.Array
        Randoms weights, shape ``(n_r,)``.
    amr : AMR_args
        AMR arguments (precomputed with :py:func:`desiwinds.forward.prepare_AMR`).
    ric : RIC_args | None
        RIC arguments of the RIC applied to the randoms before AMR, ``None`` if none (then the randoms weights do not move).

    Returns
    -------
    jax.Array
        Weights to apply to the data or the randoms (``amr.apply_to``), shape ``(n_d,)`` or ``(n_r,)``.
    """
    return apply_AMR(data_weights, randoms_weights, **{name: getattr(amr, name) for name in amr.__dataclass_fields__})


@_amr_weights.defjvp
def _amr_weights_jvp(primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
    """
    Compute the AMR weights and their tangent, the JVP rule of :py:func:`_amr_weights`.

    Parameters
    ----------
    primals : tuple
        The arguments ``(data_weights, randoms_weights, amr, ric)`` of :py:func:`_amr_weights`.
    tangents : tuple
        Their tangents; only those of the data and randoms weights are used.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The weights and their tangent.
    """
    data_weights, randoms_weights, amr, ric = primals
    ddata_weights, drandoms_weights = tangents[0], tangents[1]
    sharding_mesh = get_sharding_mesh()
    n_regions = amr.data_regions.shape[0]
    length = (amr.n_bins + 2) * n_regions
    data_exploded = bincount_sorted_vmapped(amr.data_templates_digitized, data_weights, amr.data_isort, length, sharding_mesh)
    randoms_exploded = bincount_sorted_vmapped(amr.randoms_templates_digitized, randoms_weights, amr.randoms_isort, length, sharding_mesh)
    randoms_templates_exploded = bincount_sorted_vmapped(
        amr.randoms_templates_digitized, randoms_weights * amr.randoms_templates_normalized, amr.randoms_isort, length, sharding_mesh
    )
    data_sums = (amr.data_regions * data_weights).sum(axis=1)
    randoms_sums = (amr.randoms_regions * randoms_weights).sum(axis=1)
    # tangents of the binned sums
    ddata_exploded = bincount_sorted_vmapped(amr.data_templates_digitized, ddata_weights, amr.data_isort, length, sharding_mesh)
    ddata_sums = (amr.data_regions * ddata_weights).sum(axis=1)
    if ric is None:
        drandoms_exploded, drandoms_templates_exploded, drandoms_sums = (
            jnp.zeros_like(randoms_exploded),
            jnp.zeros_like(randoms_templates_exploded),
            jnp.zeros_like(randoms_sums),
        )
    else:
        index, nric = _ric_randoms_bins(ric)
        wbins = bincount(index, weights=randoms_weights, length=nric)
        g = bincount(index, weights=drandoms_weights, length=nric) / jnp.where(wbins != 0, wbins, 1.0)
        nsys = amr.randoms_templates_digitized.shape[0]
        # cross-tables (primal only): per template t, sums of w (and w x template q) per (RIC bin, AMR bin)
        combined = index[None, :] * length + amr.randoms_templates_digitized.astype(jnp.int32)  # (nsys, n)
        cross = jnp.stack([bincount(combined[t], weights=randoms_weights, length=nric * length) for t in range(nsys)]).reshape(nsys, nric, length)
        cross_templates = jnp.stack(
            [bincount(combined[t], weights=randoms_weights * amr.randoms_templates_normalized, length=nric * length) for t in range(nsys)]
        ).reshape(nsys, nsys, nric, length)
        cross_regions = bincount(index, weights=randoms_weights * amr.randoms_regions, length=nric)  # (regions, nric)
        drandoms_exploded = jnp.einsum("k,tkl->tl", g, cross)
        drandoms_templates_exploded = jnp.einsum("k,tqkl->tlq", g, cross_templates)
        drandoms_sums = cross_regions @ g
    coefficients, dcoefficients = jax.jvp(
        _amr_coefficients,
        (data_exploded, randoms_exploded, randoms_templates_exploded, data_sums, randoms_sums),
        (ddata_exploded, drandoms_exploded, drandoms_templates_exploded, ddata_sums, drandoms_sums),
    )
    if amr.apply_to == "data":
        factor = 1 + jnp.einsum("in, ri->rn", amr.data_templates_normalized, coefficients)
        out = (amr.data_regions / factor).sum(axis=0) + jnp.invert(amr.data_regions.any(axis=0))
        dout = -(amr.data_regions * jnp.einsum("in, ri->rn", amr.data_templates_normalized, dcoefficients) / factor**2).sum(axis=0)
    else:
        out = (amr.randoms_regions * (1 + jnp.einsum("in, ri->rn", amr.randoms_templates_normalized, coefficients))).sum(axis=0) + jnp.invert(
            amr.randoms_regions.any(axis=0)
        )
        dout = (amr.randoms_regions * jnp.einsum("in, ri->rn", amr.randoms_templates_normalized, dcoefficients)).sum(axis=0)
    return out, dout


def _apply_effects_planewave(
    data_weights: list[jax.Array],
    randoms_weights: list[jax.Array],
    *,
    ric_args: tuple,
    amr_args: tuple,
    nam_args: tuple,
    data_regions: tuple,
    randoms_regions: tuple,
) -> tuple[list[jax.Array], list[jax.Array]]:
    """
    Apply the effects as :py:func:`desiwinds.forward._apply_effects` does, with the AMR of :py:func:`_amr_weights` (cheap tangent).

    Parameters
    ----------
    data_weights : list[jax.Array]
        Data weights, concatenated over the regions; one entry per tracer (length 1 for auto, 2 for cross).
    randoms_weights : list[jax.Array]
        Randoms weights, same layout as ``data_weights``.
    ric_args, amr_args, nam_args : tuple
        Tuples (possibly empty) of precomputed effect arguments, one per tracer.
    data_regions, randoms_regions : tuple
        Tuples (possibly empty) of region masks for the renormalization, one per tracer.

    Returns
    -------
    tuple[list[jax.Array], list[jax.Array]]
        Updated data and randoms weights.
    """
    data_weights, randoms_weights = list(data_weights), list(randoms_weights)

    def ric(idx):
        """Apply RIC of tracer ``idx`` to its data or randoms weights (``apply_to``), in place in the lists of weights."""
        arg = ric_args[idx]
        weight = apply_RIC(
            data_weights=data_weights[idx],
            randoms_weights=randoms_weights[idx],
            data_regions=arg.data_regions,
            randoms_regions=arg.randoms_regions,
            data_distances_digitized=arg.data_distances_digitized,
            randoms_distances_digitized=arg.randoms_distances_digitized,
            n_bins=arg.n_bins,
            apply_to=arg.apply_to,
        )
        if arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * weight
        else:
            randoms_weights[idx] = randoms_weights[idx] * weight

    for idx in range(len(ric_args)):
        ric(idx)
    for idx, amr_arg in enumerate(amr_args):
        # the randoms move before AMR only through a RIC applied to them
        ric_before = ric_args[idx] if (ric_args and ric_args[idx].apply_to == "randoms") else None
        weight = _amr_weights(data_weights[idx], randoms_weights[idx], amr_arg, ric_before)
        if amr_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * weight
        else:
            randoms_weights[idx] = randoms_weights[idx] * weight
        if ric_args:
            ric(idx)
    for idx, nam_arg in enumerate(nam_args):
        weight = apply_NAM(
            data_weights=data_weights[idx],
            randoms_weights=randoms_weights[idx],
            data_regions=nam_arg.data_regions,
            randoms_regions=nam_arg.randoms_regions,
            data_pixels=nam_arg.data_pixels,
            randoms_pixels=nam_arg.randoms_pixels,
            invsigma2=nam_arg.invsigma2,
            nside=nam_arg.nside,
            apply_to=nam_arg.apply_to,
        )
        if nam_arg.apply_to == "data":
            data_weights[idx] = data_weights[idx] * weight
        else:
            randoms_weights[idx] = randoms_weights[idx] * weight
    for idx, (_randoms_regions, _data_regions) in enumerate(zip(randoms_regions, data_regions, strict=True)):
        global_alpha = data_weights[idx].sum() / randoms_weights[idx].sum()
        alphas = (data_weights[idx] * _data_regions).sum(axis=-1) / (randoms_weights[idx] * _randoms_regions).sum(axis=-1)
        correction = (_randoms_regions * alphas[..., None] / global_alpha).sum(axis=0) + jnp.invert(_randoms_regions.any(axis=0))
        randoms_weights[idx] = randoms_weights[idx] * correction
    return data_weights, randoms_weights


def _reattr(particles: ParticleField, weights: jax.Array, attrs: MeshAttrs | None = None) -> ParticleField:
    """
    Get ``particles`` with ``weights``, on the mesh ``attrs`` if given (same box: no exchange needed), in its precision.

    Parameters
    ----------
    particles : ParticleField
        Particles to update.
    weights : jax.Array
        New weights.
    attrs : MeshAttrs | None, optional
        Mesh to put the particles on, with positions and weights cast to its precision. Default is ``None`` (keep the mesh and precision).

    Returns
    -------
    ParticleField
        Particles with the new weights.
    """
    if attrs is None:
        return particles.clone(weights=weights)
    dtype = attrs.rdtype
    return particles.clone(positions=particles.positions.astype(dtype), weights=weights.astype(dtype), attrs=attrs)


def _select_norm(norm: jax.Array | list | tuple, nkeep: int, nk: int, nells: int) -> jax.Array | list | tuple:
    """
    Select the normalization of the first ``nkeep`` of ``nk`` observed bins, per multipole (list per ell, flat array, or scalar).

    Parameters
    ----------
    norm : jax.Array | list | tuple
        Normalization: a scalar, a flat array over multipoles and bins, or a list / tuple of those (one per output spectrum).
    nkeep : int
        Number of bins to keep.
    nk : int
        Number of observed bins.
    nells : int
        Number of observed multipoles.

    Returns
    -------
    jax.Array | list | tuple
        Normalization of the kept bins, with the same nesting as ``norm``.
    """
    if isinstance(norm, (list, tuple)):
        return type(norm)(_select_norm(n, nkeep, nk, 1) for n in norm)
    norm = jnp.asarray(norm)
    if norm.size == nk * nells and nk * nells > 1:
        return norm.reshape(nells, nk)[:, :nkeep].reshape(-1) if nells > 1 else norm[..., :nkeep]
    return norm


def _legendre2(mu: jax.Array) -> jax.Array:
    r"""Return the Legendre polynomial :math:`\mathcal{L}_2(\mu)`."""
    return 1.5 * mu**2 - 0.5


def _tangents(kvec: jax.Array, positions: jax.Array, readout: Literal["exact", "mesh"], mattrs: MeshAttrs | None = None) -> jax.Array:
    r"""
    Compute the 4 tangents :math:`(u_0 \cos, u_0 \sin, u_2 \cos, u_2 \sin)` at the data ``positions``.

    Parameters
    ----------
    kvec : jax.Array
        Wavevector, shape ``(3,)``.
    positions : jax.Array
        Positions of the data particles, shape ``(n, 3)``.
    readout : Literal["exact", "mesh"]
        ``"exact"``: plane waves evaluated at the particle positions, with the line of sight of each particle. ``"mesh"``: plane waves on the mesh, read at the particles with a compensated CIC, as the forward model does.
    mattrs : MeshAttrs | None, optional
        Mesh of the data, needed for ``readout="mesh"``. Default is ``None``.

    Returns
    -------
    jax.Array
        Tangents, shape ``(4, n)``.
    """
    knorm = jnp.sqrt(jnp.sum(kvec**2))
    khat = kvec / knorm
    if readout == "exact":
        phase = positions @ kvec
        los = positions / jnp.linalg.norm(positions, axis=-1, keepdims=True)
        l2 = _legendre2(los @ khat)
        c, s = jnp.cos(phase), jnp.sin(phase)
        return jnp.stack([c, s, l2 * c, l2 * s])
    # as the forward model: the field on the mesh (line of sight of the mesh point), read with a compensated CIC
    xvec = mattrs.xcoords(kind="position", sparse=True)
    phase = sum(kk * xx for kk, xx in zip(kvec, xvec, strict=True))
    xnorm = jnp.sqrt(sum(xx**2 for xx in xvec))
    l2 = _legendre2(sum(kk * xx for kk, xx in zip(khat, xvec, strict=True)) / jnp.where(xnorm == 0, 1.0, xnorm))
    out = []
    for value in [jnp.cos(phase), jnp.sin(phase), l2 * jnp.cos(phase), l2 * jnp.sin(phase)]:
        mesh = mattrs.create(kind="real", fill=value)
        out.append(mesh.read(positions, resampler="cic", compensate=True))
    return jnp.stack(out)


def get_planewave_responses(
    *fkp_fields: FKPField | tuple[FKPField, FKPField],
    kvec: np.ndarray,
    los: Literal["local", "x", "y", "z"] = "local",
    gic: bool = True,
    ric_args: RIC_args | tuple[RIC_args, RIC_args] | None = None,
    amr_args: AMR_args | tuple[AMR_args, AMR_args] | None = None,
    nam_args: NAM_args | tuple[NAM_args, NAM_args] | None = None,
    binner: BinMesh2SpectrumPoles,
    fkp_norms: list[jax.Array],
    estimator_weights: str | tuple[str, str] | None = None,
    data_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    randoms_regions: jax.Array | tuple[jax.Array, jax.Array] | None = None,
    fields: tuple[int, int] | None = None,
    randoms_shotnoise: bool = False,
    second_order: bool = False,
    split_mixed: bool = False,
    readout: Literal["exact", "mesh"] = "exact",
    selfpairs: jax.Array | None = None,
    whitenoise: int = 0,
    seed: int = 42,
    paint_kwargs: dict | None = None,
    output_meshsize: int | None = None,
    output_kmax: float | None = None,
    data_is_randoms: bool = False,
    amr_linear: bool = True,
    mesh_dtype: str | None = None,
    batch_size: int = 1,
) -> jax.Array:
    r"""
    Compute the responses of the estimated spectra to plane waves of the mock density, per wavevector node.

    Same survey arguments as :py:func:`desiwinds.forward.mock_survey_catalog` (FKP fields of unclustered mock data, effects, binner, normalizations, estimator weights); ``los="local"`` only for now.

    Parameters
    ----------
    *fkp_fields : FKPField | tuple[FKPField, FKPField]
        FKP fields containing data and randoms information. The data shouldn't be clustered (*i.e.* the "data" should also be randoms), but the FKP field serves to designate data and randoms amongst the original randoms. One field per desired output spectrum. Pass several **tuples** of FKP fields to compute cross-spectra. For the disjoint halves of :py:func:`desiwinds.convenience.split_into_halves` or :py:func:`halves_from_split`, pass the pair as one tuple.
    kvec : np.ndarray
        Wavevector nodes, shape ``(n_nodes, 3)``, on a half space (see :py:func:`get_planewave_nodes`, :py:func:`get_mesh_mode_nodes`).
    los : Literal["local", "x", "y", "z"], optional
        Line of sight definition. Only ``"local"`` is implemented. Default is ``"local"``.
    gic : bool, optional
        Whether to apply the global integral constraint. Setting any additional effects (RIC, AMR, NAM, regions) forces GIC. Default is True.
    ric_args : RIC_args | tuple[RIC_args, RIC_args] | None, optional
        Fixed, precomputed arguments for RIC weights computation by :py:func:`desiwinds.forward.apply_RIC`. Obtain with :py:func:`desiwinds.forward.prepare_RIC`. One per tracer for cross correlation (one per half for the halves). Default is ``None``.
    amr_args : AMR_args | tuple[AMR_args, AMR_args] | None, optional
        Fixed, precomputed arguments for AMR weights computation by :py:func:`desiwinds.forward.apply_AMR`. Obtain with :py:func:`desiwinds.forward.prepare_AMR`. One per tracer for cross correlation (one per half for the halves). Default is ``None``.
    nam_args : NAM_args | tuple[NAM_args, NAM_args] | None, optional
        Fixed, precomputed arguments for NAM weights computation by :py:func:`desiwinds.forward.apply_NAM`. Obtain with :py:func:`desiwinds.forward.prepare_NAM`. One per tracer for cross correlation (one per half for the halves). Default is ``None``.
    binner : BinMesh2SpectrumPoles
        Binning operator for the power spectrum estimation.
    fkp_norms : list[jax.Array]
        Pre-computed power spectrum norms for the FKP fields ``fkp_fields``, disregarding any future changes in weights.
    estimator_weights : str | tuple[str, str] | None, optional
        Name of the weights stored in the FKP fields' particle fields ``extra_fields`` to use as extra weight at estimation time. Can pass a pair of strings for cross correlation. Default is ``None`` (no extra weight).
    data_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the data to randoms renormalization. These can typically be provided as the ``data_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``. Default is ``None``.
    randoms_regions : jax.Array | tuple[jax.Array, jax.Array] | None, optional
        Regions for the randoms for the data to randoms renormalization. These can typically be provided as the ``randoms_regions`` attribute in ``ric_args``, ``amr_args`` or ``nam_args``. Default is ``None``.
    fields : tuple[int, int] | None, optional
        ``(0, 0)`` when the two legs of a pair are the same particles with different weights (OQE variants of one tracer), as in :py:func:`desiwinds.forward.mock_survey_catalog`: their randoms shot noise is subtracted. Default is ``None``.
    randoms_shotnoise : bool, optional
        Diagnostic option, not used in the production setting (the disjoint halves have no randoms shot-noise term). Whether to subtract the first-order second derivative of the randoms shot noise :math:`\sum_j (\alpha w_j)^2` that the estimator removes (it moves with the plane wave through :math:`\alpha` and the effects on the randoms), on the monopole rows only. At :math:`k \to 0` with the global integral constraint, this removes the flat randoms floor of the lowest bin, but a floor remains in the higher bins. Default is False: the raw response, as compared to the forward model.
    second_order : bool, optional
        Whether to include :math:`2 \hat P[F_0, Q(u, u)]`, the second-order response of the effects' normalizations times the unclustered field (in the forward-model window; see the notes, section 7). About doubles the cost. Default is False.
    split_mixed : bool, optional
        Whether to return the two mixed responses separately, :math:`R_{02}` (:math:`u_0` on the first leg, :math:`u_2` on the second) and :math:`R_{20}` (the reverse), instead of their mean. Needed for the window of a cross-spectrum with a theory :math:`A_{\ell_1 \ell_2}` where :math:`A_{02} \neq A_{20}` (see :py:func:`get_planewave_window`); the plane-parallel multipoles only fix :math:`A_{02} + A_{20}`. Same cost. Not with ``second_order``. Default is False.
    readout : Literal["exact", "mesh"], optional
        ``"exact"``: the plane waves at the particle positions (the continuous-k window). ``"mesh"``: the plane waves on the mesh of each FKP field, read with a compensated CIC as the forward model does (validation). Default is ``"exact"``.
    selfpairs : jax.Array | None, optional
        Self-pair floor, shape ``(n_spectra, n_observed)``, subtracted from each node as :math:`(1, 0, \tfrac15) \times` for :math:`(R_{00}, R_{02}, R_{22})` (their direction averages; 0 for both :math:`R_{02}` and :math:`R_{20}` with ``split_mixed``); see ``whitenoise``. Default is ``None``.
    whitenoise : int, optional
        If > 0, ignore ``kvec`` and return the self-pair floor instead: the mean over ``whitenoise`` draws of the response to unit white noise on the data, :math:`\sum_i M_{ii}`, shape ``(n_spectra, n_observed)``. The i = j terms of every plane wave (independent of k), whose integral over k diverges for local-PNG shapes; to be passed as ``selfpairs``. Default is 0.
    seed : int, optional
        Seed of the white-noise draws. Default is 42.
    paint_kwargs : dict | None, optional
        Painting of the tangent fields; default the forward model's (TSC, interlacing 3, compensated). At the observed scales of interest here (k_obs well below the Nyquist frequency), ``dict(resampler="cic", interlacing=0, compensate=True)`` is ~3x cheaper.
    output_meshsize : int | None, optional
        Paint and estimate on a mesh of this size over the same box (default: the FKP fields' mesh). Plane waves are evaluated at the particles, so the input k is not limited by it; only observed bins below its Nyquist frequency are returned. With sharding, it must be divisible by the device grid (then the particles' slabs are unchanged).
    output_kmax : float | None, optional
        Keep only observed bins with upper edge <= ``output_kmax`` (and the Nyquist frequency of the output mesh). Must be given when jitting with a traced box size. Default is ``None``.
    data_is_randoms : bool, optional
        Whether data and randoms of each FKP field are the same particles (the halves of :py:func:`desiwinds.convenience.split_into_halves`, :py:func:`halves_from_split`): paint the combined weights once instead of data and randoms separately (half the cost). Default is False.
    amr_linear : bool, optional
        Whether to use the cheap exact tangent of AMR (:py:func:`_amr_weights`) rather than differentiating through :py:func:`desiwinds.forward.apply_AMR`. Default is True.
    mesh_dtype : str | None, optional
        Precision of the painted meshes and FFTs, e.g. ``"float32"`` (half the memory traffic). The particle-side differentiation (effects, AMR regression) keeps the input precision; weights and positions are cast only for painting. Default is ``None`` (the precision of the FKP fields' mesh).
    batch_size : int, optional
        Number of nodes evaluated at once (:py:func:`jax.lax.map`); each node holds 4 tangent meshes per FKP field. Default is 1.

    Returns
    -------
    jax.Array
        Shape ``(n_nodes, n_spectra, 3, n_observed)``: :math:`R_{00}, R_{02}, R_{22}` (:math:`R_{02}` the mean of both leg orders), or with ``split_mixed`` shape ``(n_nodes, n_spectra, 4, n_observed)``: :math:`R_{00}, R_{02}, R_{20}, R_{22}`, summed over :math:`\pm\mathbf{k}`, for each output spectrum (FKP field group) and observed multipole bin (concatenated as the spectrum values). With ``whitenoise > 0``, the self-pair floor, shape ``(n_spectra, n_observed)``, instead.

    Raises
    ------
    NotImplementedError
        If ``los`` is not ``"local"``, or if both ``split_mixed`` and ``second_order`` are set.
    ValueError
        If the box size is traced (under jit) and ``output_kmax`` is not given.
    """
    if los != "local":
        raise NotImplementedError("only los='local' for now")
    if split_mixed and second_order:
        # Q of each leg is quadratic in that tracer's own density: its coefficients are the tracer's auto ones, not A_{l1 l2} of the cross
        raise NotImplementedError("split_mixed is not implemented with second_order")
    if (not gic) and any(x is not None for x in (ric_args, amr_args, nam_args, data_regions, randoms_regions)):
        gic = True

    as_tuple = lambda x: () if x is None else (x if isinstance(x, tuple) else (x,))  # noqa: E731

    ric_args, amr_args, nam_args = as_tuple(ric_args), as_tuple(amr_args), as_tuple(nam_args)
    data_regions, randoms_regions = as_tuple(data_regions), as_tuple(randoms_regions)
    fkp_fields = tuple(fkp_field if isinstance(fkp_field, tuple) else (fkp_field,) for fkp_field in fkp_fields)
    _check_effects_per_field(fkp_fields, ric_args=ric_args, amr_args=amr_args, nam_args=nam_args, data_regions=data_regions, randoms_regions=randoms_regions)
    sharding_mesh = get_sharding_mesh()

    is_fkp = lambda x: isinstance(x, FKPField)  # noqa: E731

    paint_kwargs = dict(PAINT_KWARGS) if paint_kwargs is None else dict(paint_kwargs, out="real")
    ells = list(binner.ells)

    def output_attrs(input_attrs):
        """Get the mesh of the painted tangents for a field of mesh ``input_attrs``: its box (and centre), with ``output_meshsize`` / ``mesh_dtype``; None: the input mesh."""
        if output_meshsize is None:
            return None if mesh_dtype is None else input_attrs.clone(dtype=mesh_dtype)
        return get_mesh_attrs(boxsize=input_attrs.boxsize, meshsize=output_meshsize, boxcenter=input_attrs.boxcenter, dtype=mesh_dtype)

    # per field: field groups (e.g. the caps of one survey, whose effects are prepared jointly) keep their own box centre
    out_attrs_fields = tuple(tuple(output_attrs(fkp.attrs) for fkp in group) for group in fkp_fields)
    out_attrs = out_attrs_fields[0][0]
    if output_meshsize is not None or output_kmax is not None:
        if out_attrs is None:
            out_attrs = fkp_fields[0][0].attrs
        edges = np.asarray(binner.edges)
        try:
            knyq = float(np.pi * np.min(np.asarray(out_attrs.meshsize) / np.asarray(out_attrs.boxsize)))
        except jax.errors.TracerArrayConversionError:  # box size traced under jit: rely on output_kmax
            if output_kmax is None:
                raise ValueError("under jit, give output_kmax (below the Nyquist frequency of the output mesh)") from None
            knyq = np.inf
        nkeep = int(np.sum(edges[:, 1] <= min(knyq, np.inf if output_kmax is None else output_kmax)))
        binner = BinMesh2SpectrumPoles(out_attrs, edges=edges[:nkeep], ells=ells)
        fkp_norms = [_select_norm(norm, nkeep, edges.shape[0], len(ells)) for norm in fkp_norms]
    alphas_nogic = None if gic else jax.tree.map(lambda fkp: fkp.data.weights.sum() / fkp.randoms.weights.sum(), fkp_fields, is_leaf=is_fkp)

    def final_weights(deltas):
        """
        Compute final data weights and alpha-scaled randoms weights (per output spectrum, per tracer) for data densities ``deltas``.

        The painted FKP field is paint(data, w_d) - paint(randoms, alpha w_r), linear in these weights at fixed positions: only the weights are differentiated (no FFT, which jaxdecomp cannot differentiate twice), then each derivative is painted.
        """
        fields = jax.tree.map(lambda fkp, delta: fkp.clone(data=fkp.data.clone(weights=fkp.data.weights * (1 + delta))), fkp_fields, deltas, is_leaf=is_fkp)
        data_weights = [local_concatenate([f.data.weights for f in group], axis=0, sharding_mesh=sharding_mesh) for group in zip(*fields, strict=True)]
        randoms_weights = [local_concatenate([f.randoms.weights for f in group], axis=0, sharding_mesh=sharding_mesh) for group in zip(*fields, strict=True)]
        apply_effects = _apply_effects_planewave if amr_linear else _apply_effects
        data_weights, randoms_weights = apply_effects(
            data_weights, randoms_weights, ric_args=ric_args, amr_args=amr_args, nam_args=nam_args, data_regions=data_regions, randoms_regions=randoms_regions
        )
        data_weights, randoms_weights = _split_weights(data_weights, randoms_weights, fields, sharding_mesh=sharding_mesh)
        fields = jax.tree.map(_update_fkp, data_weights, randoms_weights, fields, _fill_with_constant(data_weights, estimator_weights))
        if gic:  # as FKPField.paint: alpha = sum of data weights / sum of randoms weights
            alphas = jax.tree.map(lambda f: f.data.weights.sum() / f.randoms.weights.sum(), fields, is_leaf=is_fkp)
        else:
            alphas = alphas_nogic
        return jax.tree.map(lambda f, alpha: (f.data.weights, alpha * f.randoms.weights), fields, alphas, is_leaf=is_fkp)

    def paint(weights):
        """Paint meshes (per output spectrum, per tracer) for (data, randoms) weights: data - randoms."""
        if data_is_randoms:
            return tuple(
                tuple(_reattr(fkp.data, wd - wr, oattrs).paint(**paint_kwargs).value for fkp, (wd, wr), oattrs in zip(group, wgroup, ogroup, strict=True))
                for group, wgroup, ogroup in zip(fkp_fields, weights, out_attrs_fields, strict=True)
            )
        return tuple(
            tuple(
                (_reattr(fkp.data, wd, oattrs).paint(**paint_kwargs) - _reattr(fkp.randoms, wr, oattrs).paint(**paint_kwargs)).value
                for fkp, (wd, wr), oattrs in zip(group, wgroup, ogroup, strict=True)
            )
            for group, wgroup, ogroup in zip(fkp_fields, weights, out_attrs_fields, strict=True)
        )

    zeros = jax.tree.map(lambda fkp: jnp.zeros_like(fkp.data.weights), fkp_fields, is_leaf=is_fkp)
    primal = paint(final_weights(zeros))
    # randoms shot noise: same particles in both legs (auto, or OQE variants with fields=(0, 0)); 1 / norm_0 on the monopole rows
    same_particles = [len(group) == 1 or fields == (0, 0) for group in fkp_fields]
    nk = [int(np.size(binner.xavg))] * len(ells)  # the same k bins for all multipoles
    monopole = [
        jnp.concatenate([jnp.full(nk[ill], float(ell == 0)) for ill, ell in enumerate(ells)])
        / jnp.ravel(jnp.asarray(fkp_norms[iout][ells.index(0)] if 0 in ells else jnp.inf))[0]
        for iout in range(len(fkp_fields))
    ]
    attrs = [[f.attrs if o is None else o for f, o in zip(group, ogroup, strict=True)] for group, ogroup in zip(fkp_fields, out_attrs_fields, strict=True)]

    def spectrum(mesh1, mesh2, attrs1, iout):
        """Compute the raw (no shot noise subtracted) cross-spectrum of the real meshes ``mesh1`` and ``mesh2`` of the output spectrum ``iout``, on the mesh ``attrs1``."""
        pk = compute_mesh2_spectrum(attrs1.create(kind="real", fill=mesh1), attrs1.create(kind="real", fill=mesh2), bin=binner, los="firstpoint")
        pk = pk.clone(norm=fkp_norms[iout])  # no shot-noise subtraction: the raw response
        return jnp.concatenate(pk.value(concatenate=False)).real

    def tangents_of(k):
        """Per FKP field, the 4 tangents at its data; nested as fkp_fields, leading axis 4."""
        return jax.tree.map(lambda fkp: _tangents(k, fkp.data.positions, readout, mattrs=fkp.attrs), fkp_fields, is_leaf=is_fkp)

    # linearized once: the primal (and everything only it depends on, e.g. the AMR cross-tables) is not recomputed per node
    _, first = jax.linearize(final_weights, zeros)

    def second(u):
        """Second derivative of the weights along u (= 2 Q(u, u) once painted)."""
        return jax.jvp(lambda d: jax.jvp(final_weights, (d,), (u,))[1], (zeros,), (u,))[1]

    def node(k):
        r"""Compute the responses :math:`(R_{00}, R_{02}, R_{22})` of all the output spectra to the plane wave of wavevector ``k``, summed over :math:`\pm k`."""
        u = tangents_of(k)  # leaves (4, n): u0c, u0s, u2c, u2s

        pick = lambda tree, i: jax.tree.map(lambda t: t[i], tree)  # noqa: E731

        # weight derivatives: particle arrays, vmapped over the tangents; painting (FFTs, sharded with jaxdecomp, which takes a
        # single batch dimension, used by lax.map over nodes) is unrolled
        dweights = jax.vmap(first)(u)  # leaves (4, n): (data, alpha x randoms) weight derivatives
        lin = [paint(pick(dweights, i)) for i in range(4)]
        lin = jax.tree.map(lambda *leaves: jnp.stack(leaves), *lin)  # leaves (4, *meshshape)
        if second_order:
            # Q(u0, u0), Q(u0s, u0s), Q(u2, u2), Q(u2s, u2s), Q(u0 + u2, ...) for cos and sin; halves applied below
            sums = jax.tree.map(lambda t: jnp.stack([t[0] + t[2], t[1] + t[3]]), u)
            d2 = jax.vmap(second)(jax.tree.map(lambda a, b: jnp.concatenate([a, b]), u, sums))
            quad = [paint(pick(d2, i)) for i in range(6)]

        out = []
        for iout, group in enumerate(attrs):
            t1, t2 = (0, 0) if len(group) == 1 else (0, 1)
            a1 = group[0]
            m = lambda leaf_index, tracer, iout=iout: lin[iout][tracer][leaf_index]  # noqa: E731
            r00 = r02 = r20 = r22 = 0.0
            for i0, i2 in [(0, 2), (1, 3)]:  # (u0, u2) indices for cos, then sin
                r00 = r00 + spectrum(m(i0, t1), m(i0, t2), a1, iout)
                r02 = r02 + spectrum(m(i0, t1), m(i2, t2), a1, iout)  # u0 on the first leg, u2 on the second: A_02
                r20 = r20 + spectrum(m(i2, t1), m(i0, t2), a1, iout)  # the reverse: A_20
                r22 = r22 + spectrum(m(i2, t1), m(i2, t2), a1, iout)
            if randoms_shotnoise and same_particles[iout]:
                # the estimator subtracts the randoms shot noise sum (alpha w_r)^2 (monopole, flat), which moves with the
                # plane wave through alpha and the effects on the randoms: subtract its second derivative as the FM does
                wr = lambda leaf_index, tracer, iout=iout: dweights[iout][tracer][1][leaf_index]  # noqa: E731
                for i0, i2 in [(0, 2), (1, 3)]:
                    r00 = r00 - monopole[iout] * jnp.sum(wr(i0, t1) * wr(i0, t2))
                    r02 = r02 - monopole[iout] * jnp.sum(wr(i0, t1) * wr(i2, t2))
                    r20 = r20 - monopole[iout] * jnp.sum(wr(i2, t1) * wr(i0, t2))
                    r22 = r22 - monopole[iout] * jnp.sum(wr(i2, t1) * wr(i2, t2))
            mixed = [r02, r20] if split_mixed else [0.5 * (r02 + r20)]
            r = 2.0 * jnp.stack([r00, *mixed, r22])  # + k and - k
            if second_order:
                f0 = primal[iout]
                for q0, q2, q02 in [(0, 2, 4), (1, 3, 5)]:
                    # Q(u, v) for the pairs (0,0), (0,2), (2,2); Q(u, u) = second / 2
                    qa = [0.5 * quad[q0][iout][t] for t in range(len(group))]
                    qb = [0.5 * quad[q2][iout][t] for t in range(len(group))]
                    qab = [0.5 * (0.5 * quad[q02][iout][t] - qa[t] - qb[t]) for t in range(len(group))]
                    for idx, q in enumerate([qa, qab, qb]):
                        # 2 P[F0, Q] symmetrized over the two tracers, x 2 for + k and - k
                        term = spectrum(f0[t1], q[t2], a1, iout) + spectrum(q[t1], f0[t2], a1, iout)
                        r = r.at[idx].add(2.0 * term)
            out.append(r)
        return jnp.stack(out)

    if whitenoise:

        def draw(key):
            """Compute the response of all the output spectra to unit white noise on the data, for the random key ``key``."""
            keys = jax.random.split(key, len(jax.tree.leaves(zeros)))
            eps = jax.tree.unflatten(
                jax.tree.structure(zeros), [jax.random.normal(kk, z.shape, dtype=z.dtype) for kk, z in zip(keys, jax.tree.leaves(zeros), strict=True)]
            )
            mesh = paint(first(eps))
            return jnp.stack([spectrum(mesh[iout][0], mesh[iout][-1], group[0], iout) for iout, group in enumerate(attrs)])

        return jnp.mean(jax.lax.map(draw, jax.random.split(jax.random.key(seed), whitenoise), batch_size=batch_size), axis=0)
    responses = jax.lax.map(node, jnp.asarray(kvec), batch_size=batch_size)
    if selfpairs is not None:
        floor = [1.0, 0.0, 0.0, 0.2] if split_mixed else [1.0, 0.0, 0.2]  # direction averages of (R00, R02, [R20,] R22)
        responses = responses - 2.0 * jnp.asarray(floor)[None, None, :, None] * jnp.asarray(selfpairs)[None, :, None, :]  # x 2: + k and - k
    return responses


def get_planewave_window(responses: jax.Array, weights: np.ndarray, kvec: np.ndarray, shapes: Callable[[np.ndarray], np.ndarray]) -> jax.Array:
    r"""
    Get the window columns from node responses: :math:`\sum_n \omega_n \sum_\ell V_\ell(k_n) \mathcal{R}_\ell(\mathbf{k}_n)`.

    Parameters
    ----------
    responses : jax.Array
        From :py:func:`get_planewave_responses`, shape ``(n_nodes, n_spectra, 3, n_observed)``, or ``(n_nodes, n_spectra, 4, n_observed)`` with ``split_mixed=True``.
    weights : np.ndarray
        Quadrature weights of the nodes.
    kvec : np.ndarray
        The nodes, shape ``(n_nodes, 3)``.
    shapes : Callable[[np.ndarray], np.ndarray]
        ``shapes(k) -> array (n_columns, 3, len(k))``: the theory multipoles :math:`P_0, P_2, P_4` of each column at ``k`` (a basis vector, or 1 in a bin and 0 elsewhere for a spike). Or ``(n_columns, 4, len(k))``: the theory :math:`A_{00}, A_{02}, A_{20}, A_{22}` of a cross-spectrum with the local line of sight (the convention of :py:func:`jaxpower.generate_anisotropic_gaussian_mesh`: :math:`\ell_1` on the first leg), which needs the responses of ``split_mixed=True`` unless :math:`A_{02} = A_{20}`.

    Returns
    -------
    jax.Array
        Shape ``(n_spectra, n_observed, n_columns)``.

    Raises
    ------
    ValueError
        If ``shapes`` gives :math:`A_{02} \neq A_{20}` with responses of the mean of both mixed orders, or has neither 3 nor 4 multipoles.
    """
    knorm = np.linalg.norm(np.asarray(kvec), axis=-1)
    p = np.asarray(shapes(knorm))  # (n_columns, 3 or 4, n_nodes)
    if p.shape[1] == 3:
        # (c_aa, c_ab = c_ba, c_bb) linear in (P0, P2, P4)
        p0, p2, p4 = p[:, 0], p[:, 1], p[:, 2]
        cbb = 35.0 / 18.0 * p4
        cab = cba = 0.5 * p2 - cbb / 7.0
        caa = p0 - cbb / 5.0
    elif p.shape[1] == 4:
        caa, cab, cba, cbb = (p[:, i] for i in range(4))  # A00, A02, A20, A22
    else:
        raise ValueError(f"shapes must give 3 multipoles (P0, P2, P4) or 4 (A00, A02, A20, A22), found {p.shape[1]}")
    if responses.shape[2] == 3:
        if not np.allclose(cab, cba, rtol=1e-12, atol=0.0):
            raise ValueError("A02 != A20 needs the responses of get_planewave_responses(..., split_mixed=True)")
        mixed = [cab + cba]  # times the mean of both mixed orders
    else:
        mixed = [cab, cba]
    coeffs = np.stack([caa, *mixed, cbb], axis=1) * np.asarray(weights)[None, None, :]  # (n_columns, 3 or 4, n_nodes)
    # n: nodes, s: output spectra, r: (R00, R02, [R20,] R22), o: observed bins, c: columns
    return jnp.einsum("nsro,crn->soc", responses, coeffs)
