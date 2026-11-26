"""
Tests for :py:mod:`desiwinds.planewave` on the small synthetic survey of ``test_window_basis.py`` (CPU, a few minutes).
"""

import jax

jax.config.update("jax_enable_x64", True)

import lsstypes as lss
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpower import BinMesh2SpectrumPoles, FKPField, ParticleField, compute_fkp2_normalization, get_mesh_attrs

from desiwinds.convenience import split_into_halves
from desiwinds.forward import mock_survey_catalog
from desiwinds.planewave import get_mesh_mode_nodes, get_planewave_nodes, get_planewave_responses, get_planewave_window
from desiwinds.window import get_window_spikes

N_PARTICLES = 20000
NREAL = 64
BIN = (0.04, 0.06)


def _ball(rng, n):
    x = rng.normal(size=(n, 3))
    x *= (150 * rng.uniform(size=n) ** (1 / 3) / np.linalg.norm(x, axis=1))[:, None]
    return x + np.array([800.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def survey():
    rng = np.random.default_rng(42)
    data_positions, randoms_positions = _ball(rng, N_PARTICLES), _ball(rng, N_PARTICLES)
    mattrs = get_mesh_attrs(data_positions, randoms_positions, cellsize=20.0, check=True)
    fkp = FKPField(ParticleField(data_positions, np.ones(N_PARTICLES), attrs=mattrs), ParticleField(randoms_positions, np.ones(N_PARTICLES), attrs=mattrs), attrs=mattrs)
    binner = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.02}, ells=(0, 2, 4))
    norm = compute_fkp2_normalization(fkp, bin=binner, cellsize=20.0)
    # all particles as both data and randoms: no sampling noise of a data / randoms split
    all_positions = np.concatenate([data_positions, randoms_positions])
    fkp_nosplit = FKPField(ParticleField(all_positions, np.ones(2 * N_PARTICLES), attrs=mattrs), ParticleField(all_positions, np.ones(2 * N_PARTICLES), attrs=mattrs), attrs=mattrs)
    norm_nosplit = compute_fkp2_normalization(fkp_nosplit, bin=binner, cellsize=20.0)
    return dict(mattrs=mattrs, fkp=fkp, binner=binner, norm=norm, fkp_nosplit=fkp_nosplit, norm_nosplit=norm_nosplit)


def _halves(fkp, seed):
    """Two disjoint halves of the pooled data and randoms of ``fkp``."""
    positions = np.concatenate([np.asarray(fkp.data.positions), np.asarray(fkp.randoms.positions)])
    weights = np.concatenate([np.asarray(fkp.data.weights), np.asarray(fkp.randoms.weights)])
    return split_into_halves(positions, weights, None, split_seed=seed, mattrs=fkp.attrs)


def _bin_shapes(k):
    inbin = ((k >= BIN[0]) & (k < BIN[1])).astype(float)
    zero = np.zeros_like(inbin)
    return np.array([[inbin, zero, zero], [zero, inbin, zero], [zero, zero, inbin]])


def test_mesh_modes_equal_forward_model(survey):
    """Plane waves on the mesh modes of a theory bin, read as the FM does = the seed-averaged FM spike column (raw)."""
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    edges = np.arange(0.0, 0.16, 0.02)
    k = (edges[:-1] + edges[1:]) / 2
    p0 = 2e4 * np.exp(-k / 0.1)
    theory = lss.Mesh2SpectrumPoles(
        [lss.Mesh2SpectrumPole(k=k, k_edges=np.stack([edges[:-1], edges[1:]], -1), num_raw=f * p0, ell=ell) for ell, f in [(0, 1.0), (2, 0.3), (4, 0.05)]], ells=[0, 2, 4]
    )

    def mock_survey(theory, seed, return_data_shotnoise=False):
        return mock_survey_catalog(fkp, theory=theory, seed=seed, los="local", unitary_amplitude=True, ric_args=None, amr_args=None, nam_args=None,
                                   binner=binner, fkp_norms=[norm], estimator_weights=None, return_data_shotnoise=return_data_shotnoise)

    _, reals = get_window_spikes(mock_survey, theory, nreal=NREAL, batch_size=8, return_shotnoise_jacobians=True)
    raw = np.array([np.asarray(w[0].value()) + np.asarray(s[0].value()) for w, s in zip(reals["window"], reals["shotnoise_jacobian"], strict=True)])
    kvec, weights = get_mesh_mode_nodes(survey["mattrs"], BIN)
    responses = get_planewave_responses(fkp, kvec=kvec, binner=binner, fkp_norms=[norm], second_order=True, readout="mesh", batch_size=8)
    window = np.asarray(get_planewave_window(responses, weights, kvec, _bin_shapes))[0]
    ibin = int(np.searchsorted(edges, BIN[0]))
    for icol in range(3):
        fm = raw[:, :, ibin + icol * len(k)]
        mean, error = fm.mean(axis=0), fm.std(axis=0) / np.sqrt(NREAL)
        pull = (window[:, icol] - mean) / np.where(error > 0, error, np.inf)
        assert np.median(np.abs(pull)) < 1.5 and np.abs(pull).max() < 5, (icol, pull)


def _two_tracers(n=10000):
    """Two tracers in overlapping balls (data / randoms split each), as in test_cross_pairs.py."""
    rng = np.random.default_rng(7)

    def ball(center):
        x = rng.normal(size=(n, 3))
        x *= (150 * rng.uniform(size=n) ** (1 / 3) / np.linalg.norm(x, axis=1))[:, None]
        return x + np.array(center)

    positions = [(ball(center), ball(center)) for center in ([800.0, 0.0, 0.0], [820.0, 20.0, 0.0])]
    mattrs = get_mesh_attrs(*[p for pair in positions for p in pair], cellsize=20.0, check=True)
    fkps = tuple(FKPField(ParticleField(d, np.ones(n), attrs=mattrs), ParticleField(r, np.ones(n), attrs=mattrs), attrs=mattrs) for d, r in positions)
    binner = BinMesh2SpectrumPoles(mattrs, edges={"step": 0.02}, ells=(0, 2, 4))
    norm = compute_fkp2_normalization(*fkps, bin=binner, cellsize=20.0)
    return mattrs, fkps, binner, norm


def _pair_shapes(k):
    """One spike column per (ell1, ell2) in BIN: (A00, A02, A20, A22) = identity in the bin."""
    inbin = ((k >= BIN[0]) & (k < BIN[1])).astype(float)
    return np.eye(4)[:, :, None] * inbin


def test_split_mixed_consistency():
    """split_mixed: R00, R22 unchanged, (R02 + R20) / 2 = the mean; the window from (P0, P2, P4) or the equivalent A_{l1 l2} columns agree."""
    mattrs, fkps, binner, norm = _two_tracers()
    kvec, weights = get_planewave_nodes(np.array([0.03, 0.05]), n_k=1, n_directions=4)
    kw = dict(binner=binner, fkp_norms=[norm], batch_size=4)
    mean, split = (np.asarray(get_planewave_responses(fkps, kvec=kvec, split_mixed=s, **kw)) for s in (False, True))
    assert mean.shape[2] == 3 and split.shape[2] == 4
    scale = np.abs(mean).max()
    np.testing.assert_allclose(split[:, :, [0, 3]], mean[:, :, [0, 2]], rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(0.5 * (split[:, :, 1] + split[:, :, 2]), mean[:, :, 1], rtol=0, atol=1e-12 * scale)
    assert np.abs(split[:, :, 1] - split[:, :, 2]).max() > 1e-6 * scale  # two different tracers: the orders differ

    def poles(k):  # P0, P2, P4 of a Kaiser-like column
        return np.array([[np.exp(-k / 0.05), 0.4 * np.exp(-k / 0.05), 0.1 * np.exp(-k / 0.05)]])

    def pairs(k):  # the same column as A_{l1 l2}, with A02 = A20
        p0, p2, p4 = poles(k)[0]
        cbb = 35.0 / 18.0 * p4
        cab = 0.5 * p2 - cbb / 7.0
        return np.array([[p0 - cbb / 5.0, cab, cab, cbb]])

    reference = np.asarray(get_planewave_window(mean, weights, kvec, poles))
    for responses, shapes in [(split, poles), (split, pairs), (mean, pairs)]:
        np.testing.assert_allclose(np.asarray(get_planewave_window(responses, weights, kvec, shapes)), reference, rtol=1e-10, atol=1e-12 * np.abs(reference).max())
    with pytest.raises(ValueError):  # A02 != A20 needs the split responses
        get_planewave_window(mean, weights, kvec, lambda k: pairs(k) * np.array([1.0, 2.0, 0.5, 1.0])[None, :, None])
    with pytest.raises(NotImplementedError):
        get_planewave_responses(fkps, kvec=kvec, split_mixed=True, second_order=True, **kw)


def test_mesh_modes_pairs_equal_forward_model():
    """Cross-spectrum with A02 != A20: plane waves on the mesh modes of a theory bin (split_mixed) = the seed-averaged FM spike columns."""
    try:
        from desiwinds.convenience import fiducial_theory
    except ImportError:
        pytest.skip("needs the A_{l1 l2} forward model of desiwinds main")
    mattrs, fkps, binner, norm = _two_tracers()
    edges = np.arange(0.0, 0.16, 0.02)
    theory = fiducial_theory(edges, bA=2.0, bB=1.0)  # A02 != A20
    assert [tuple(ell) for ell in theory.ells] == [(0, 0), (0, 2), (2, 0), (2, 2)]

    def mock_survey(theory, seed, return_data_shotnoise=False):
        return mock_survey_catalog(fkps, theory=theory, seed=seed, los="local", unitary_amplitude=True, ric_args=None, amr_args=None, nam_args=None,
                                   binner=binner, fkp_norms=[norm], estimator_weights=None, return_data_shotnoise=return_data_shotnoise)

    _, reals = get_window_spikes(mock_survey, theory, nreal=NREAL, batch_size=8, return_shotnoise_jacobians=True)
    raw = np.array([np.asarray(w[0].value()) + np.asarray(s[0].value()) for w, s in zip(reals["window"], reals["shotnoise_jacobian"], strict=True)])
    kvec, weights = get_mesh_mode_nodes(mattrs, BIN)
    # GIC only: the effects are linear in the density (no Q term), so the first order is the whole FM window
    responses = get_planewave_responses(fkps, kvec=kvec, binner=binner, fkp_norms=[norm], split_mixed=True, readout="mesh", batch_size=8)
    window = np.asarray(get_planewave_window(responses, weights, kvec, _pair_shapes))[0]
    nk = len(edges) - 1
    ibin = int(np.searchsorted(edges, BIN[0]))
    for icol in range(4):
        fm = raw[:, :, ibin + icol * nk]
        mean, error = fm.mean(axis=0), fm.std(axis=0) / np.sqrt(NREAL)
        pull = (window[:, icol] - mean) / np.where(error > 0, error, np.inf)
        assert np.median(np.abs(pull)) < 1.5 and np.abs(pull).max() < 5, (icol, pull)
    # and the mixed columns differ: the FM distinguishes A02 from A20, as the split responses do
    assert not np.allclose(window[:, 1], window[:, 2], rtol=1e-3)


def test_direction_convergence(survey):
    """The continuous quadrature has converged in directions at these scales, for the observed monopole and quadrupole."""
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    windows = []
    for n_directions in (32, 64):
        kvec, weights = get_planewave_nodes(np.linspace(*BIN, 3), n_k=4, n_directions=n_directions)
        responses = get_planewave_responses(fkp, kvec=kvec, binner=binner, fkp_norms=[norm], batch_size=8)
        windows.append(np.asarray(get_planewave_window(responses, weights, kvec, _bin_shapes))[0, :, 0])
    # observed ell = 0, 2 (the observed hexadecapole needs more directions: 7% between 32 and 64 here)
    nk = windows[1].size // 3
    assert np.allclose(windows[0][:2 * nk], windows[1][:2 * nk], rtol=0, atol=1e-2 * np.abs(windows[1]).max())


def test_gic_kills_uniform_mode(survey):
    """Without a data / randoms split, the response to plane waves vanishes as k -> 0 with the global integral constraint, not without."""
    binner = survey["binner"]
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.009 * kf, 0.011 * kf, 1.3 * kf, 1.5 * kf]), n_k=1, n_directions=8)
    knorm = np.linalg.norm(kvec, axis=-1)
    low, high = knorm < 0.1 * kf, knorm > kf
    r = {}
    for gic in (True, False):
        r[gic] = np.asarray(get_planewave_responses(survey["fkp_nosplit"], kvec=kvec, gic=gic, binner=binner, fkp_norms=[survey["norm_nosplit"]], batch_size=8))[:, 0, 0]
    nk = r[True].shape[-1] // 3
    gic_low, gic_high, nogic_low = (np.abs(x[:, :nk]).mean(axis=0) for x in (r[True][low], r[True][high], r[False][low]))
    assert np.all(gic_low < 1e-3 * nogic_low)
    assert np.all(gic_low[:2] < 1e-2 * gic_high[:2])


def test_randoms_shotnoise_diagnostic(survey):
    """``randoms_shotnoise`` (diagnostic) only changes the monopole rows, and removes the flat randoms floor of the lowest bin as k -> 0."""
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.009 * kf, 0.011 * kf]), n_k=1, n_directions=8)
    r = {sn: np.asarray(get_planewave_responses(fkp, kvec=kvec, binner=binner, fkp_norms=[norm], randoms_shotnoise=sn, batch_size=8))[:, 0, 0] for sn in (False, True)}
    nk = r[True].shape[-1] // 3
    np.testing.assert_array_equal(r[True][:, nk:], r[False][:, nk:])  # no change of the ell = 2, 4 rows
    raw, subtracted = (np.abs(x[:, 0]).mean() for x in (r[False], r[True]))
    assert subtracted < 0.1 * raw  # lowest bin; the higher bins keep a floor (not asserted)


def test_halves_cross_has_no_selfpairs(survey):
    """
    The cross-response of two disjoint halves (each data = randoms) has no self-pair term: it does not depend on the split, equals
    the raw response where the self-pair part vanishes (kR << 1), and is below it where the self pairs are a white noise (kR >> 1).
    """
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.04 * kf, 0.06 * kf, 3.9 * kf, 4.1 * kf]), n_k=1, n_directions=16)
    knorm = np.linalg.norm(kvec, axis=-1)
    low, high = knorm < 0.1 * kf, knorm > 2 * kf  # kR ~ 0.08 and ~ 6 (R = 150)
    nk = int(np.size(binner.xavg))

    def average(responses):
        responses = np.asarray(responses)[:, 0, 0]
        return responses[low].mean(axis=0)[:nk], responses[high].mean(axis=0)[:nk]

    cross = [average(get_planewave_responses(_halves(fkp, seed), kvec=kvec, binner=binner, fkp_norms=[norm], batch_size=8)) for seed in (1, 2)]
    # the full pool as data = randoms, with the same total data weight (same normalization)
    pool = survey["fkp_nosplit"]
    pool = FKPField(pool.data.clone(weights=pool.data.weights / 2), pool.randoms.clone(weights=pool.randoms.weights / 2), attrs=pool.attrs)
    raw = average(get_planewave_responses(pool, kvec=kvec, binner=binner, fkp_norms=[norm], batch_size=8))
    from desiwinds.planewave import halves_from_split

    halves, ric, amr = halves_from_split(fkp)  # the FM's own data / randoms split as the two halves
    assert ric is None and amr is None  # as prepare_halves_effects: None when not given
    cross.append(average(get_planewave_responses(halves, kvec=kvec, binner=binner, fkp_norms=[norm], batch_size=8)))
    for other in cross[1:]:
        for i in range(2):
            assert np.allclose(cross[0][i][:5], other[i][:5], rtol=5e-2)
    # where the window dominates; in the smaller bins the self-pair part is small but not zero (S ~ 0.002 T D at kR = 0.08)
    assert np.allclose(cross[0][0][:3], raw[0][:3], rtol=2e-2)
    # at high k, the raw response has the extra self-pair white noise in the observed bins where the window is small
    assert np.all(raw[1][3:5] > cross[0][1][3:5])


def test_prepare_halves_effects(survey):
    """Effect arguments prepared natively on the halves (RIC) are those derived from the FM split, and give the same responses."""
    from desiwinds.forward import prepare_RIC
    from desiwinds.planewave import halves_from_split, prepare_halves_effects

    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    ric_kwargs = dict(regions=["ALL"], n_bins=10, apply_to="randoms")
    halves, ric_fm, _ = halves_from_split(fkp, prepare_RIC(data=[fkp.data], randoms=[fkp.randoms], **ric_kwargs))
    ric, amr, nam = prepare_halves_effects(halves, ric_kwargs=ric_kwargs)
    assert amr is None and nam is None
    for native, shortcut in zip(ric, ric_fm, strict=True):
        assert not np.asarray(native.data_to_remove).any()
        for name in ("data_distances_digitized", "randoms_distances_digitized", "data_regions", "randoms_regions"):
            np.testing.assert_array_equal(np.asarray(getattr(native, name)), np.asarray(getattr(shortcut, name)))
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.5 * kf, 2 * kf]), n_k=1, n_directions=4)

    def respond(args):
        return np.asarray(
            get_planewave_responses(halves, kvec=kvec, ric_args=args, data_regions=tuple(a.data_regions for a in args), randoms_regions=tuple(a.randoms_regions for a in args),
                                    binner=binner, fkp_norms=[norm], data_is_randoms=True, batch_size=4)
        )

    reference = respond(ric_fm)
    assert np.abs(reference).max() > 0
    np.testing.assert_allclose(respond(ric), reference, rtol=1e-10, atol=1e-10 * np.abs(reference).max())


def test_prepare_halves_effects_amr(survey):
    """RIC + AMR prepared natively on the halves (own template bins) agree with those derived from the FM split to ~1e-4 of the response."""
    from desiwinds.forward import prepare_AMR, prepare_RIC
    from desiwinds.planewave import halves_from_split, prepare_halves_effects

    rng = np.random.default_rng(3)

    def with_templates(particles):
        n = particles.size
        return particles.clone(extra={"Z": rng.uniform(0.8, 1.2, n), "template_values": rng.normal(size=(n, 2)) + 0.3 * np.asarray(particles.positions)[:, :1] / 150})

    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    fkp = FKPField(with_templates(fkp.data), with_templates(fkp.randoms), attrs=fkp.attrs)
    ric_kwargs = dict(regions=["ALL"], n_bins=10, apply_to="randoms")
    amr_kwargs = dict(regions_zranges=[("ALL", (0.0, 2.0))], apply_to="randoms", n_bins=5)
    halves, ric_fm, amr_fm = halves_from_split(
        fkp, prepare_RIC(data=[fkp.data], randoms=[fkp.randoms], **ric_kwargs), prepare_AMR(data=[fkp.data], randoms=[fkp.randoms], keep_data_templates_normalized=True, **amr_kwargs)
    )
    ric, amr, _ = prepare_halves_effects(halves, ric_kwargs=ric_kwargs, amr_kwargs=amr_kwargs)
    assert all(a.data_templates_normalized is not None for a in amr)  # data used as randoms
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.5 * kf, 2 * kf]), n_k=1, n_directions=4)

    def respond(ric_args, amr_args):
        return np.asarray(
            get_planewave_responses(halves, kvec=kvec, ric_args=ric_args, amr_args=amr_args, data_regions=tuple(a.data_regions for a in ric_args),
                                    randoms_regions=tuple(a.randoms_regions for a in ric_args), binner=binner, fkp_norms=[norm], data_is_randoms=True, batch_size=4)
        )

    reference, native = respond(ric_fm, amr_fm), respond(ric, amr)
    assert np.isfinite(native).all()
    assert np.abs(native - reference).max() < 1e-3 * np.abs(reference).max()


def test_split_into_halves(survey):
    """The halves partition the particles (sizes, total weight, extra columns kept with their particle), each used as data and randoms."""
    fkp = survey["fkp"]
    positions = np.concatenate([np.asarray(fkp.data.positions), np.asarray(fkp.randoms.positions)])
    weights = np.concatenate([np.asarray(fkp.data.weights), np.asarray(fkp.randoms.weights)])
    halves = split_into_halves(positions, weights, {"Z": positions[:, 0]}, split_seed=5, mattrs=fkp.attrs)
    assert sum(half.data.size for half in halves) == positions.shape[0]
    for half in halves:  # same particles as data and randoms
        np.testing.assert_array_equal(np.asarray(half.data.positions), np.asarray(half.randoms.positions))
    assert abs(halves[0].data.size / positions.shape[0] - 0.5) < 0.02
    assert np.isclose(sum(np.asarray(half.data.weights).sum() for half in halves), weights.sum())
    for half in halves:
        np.testing.assert_array_equal(np.asarray(half.data.extra["Z"]), np.asarray(half.data.positions)[:, 0])
    assert not np.array_equal(np.sort(np.asarray(halves[0].data.positions)[:, 0])[:50], np.sort(np.asarray(halves[1].data.positions)[:, 0])[:50])


def test_cheap_options(survey):
    """data_is_randoms (one paint per half) is identical; CIC without interlacing on a coarser output mesh agrees at low k_obs."""
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    halves = _halves(fkp, 1)
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.5 * kf, 2 * kf]), n_k=1, n_directions=4)
    reference = np.asarray(get_planewave_responses(halves, kvec=kvec, binner=binner, fkp_norms=[norm], batch_size=4))
    once = np.asarray(get_planewave_responses(halves, kvec=kvec, binner=binner, fkp_norms=[norm], data_is_randoms=True, batch_size=4))
    assert np.allclose(once, reference, rtol=1e-10, atol=1e-10 * np.abs(reference).max())
    cheap = np.asarray(
        get_planewave_responses(halves, kvec=kvec, binner=binner, fkp_norms=[norm], data_is_randoms=True, batch_size=4,
                                paint_kwargs=dict(resampler="cic", interlacing=0, compensate=True), output_meshsize=16, output_kmax=0.05)
    )
    nk, nkc = reference.shape[-1] // 3, cheap.shape[-1] // 3
    # observed monopole bins kept on the 16^3 mesh, except bin 0 (the k = 0 mode only)
    a, b = cheap[..., 0, 1:nkc], reference[..., 0, 1:nkc]
    assert np.allclose(a, b, rtol=1e-2)


@pytest.mark.parametrize("apply_to", ["randoms", "data"])
def test_amr_linear_tangent(apply_to):
    """The cheap AMR tangent (cross-tables of RIC and AMR bins) equals autodiff through apply_AMR, with RIC before it."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_forward_effects import _wrap, make_tracer
    from desiwinds.forward import _apply_effects
    from desiwinds.planewave import _apply_effects_planewave

    tracer = make_tracer(3, apply_to=("randoms", apply_to, "randoms"))
    kwargs = _wrap([tracer], use_ric=True, use_amr=True, use_nam=False)
    rng = np.random.default_rng(1)
    ddata = jnp.asarray(rng.normal(size=tracer.data_weights.shape)) * tracer.data_weights

    def run(effects, delta):
        data, randoms = effects([tracer.data_weights * (1 + delta)], [tracer.randoms_weights], **kwargs)
        return data[0], randoms[0]

    zero = jnp.zeros_like(tracer.data_weights)
    primal, tangent = jax.jvp(lambda d: run(_apply_effects, d), (zero,), (ddata,))
    primal_lin, tangent_lin = jax.jvp(lambda d: run(_apply_effects_planewave, d), (zero,), (ddata,))
    for a, b in zip(primal_lin + tangent_lin, primal + tangent, strict=True):
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-8, atol=1e-10 * np.abs(np.asarray(b)).max())
    assert np.abs(np.asarray(tangent[1])).max() > 0  # the randoms do move


def test_float32_meshes(survey):
    """Painting and FFTs in float32 (weight derivatives in float64) agree with float64 to ~1e-4, down to kR ~ 1e-3."""
    fkp, binner, norm = survey["fkp"], survey["binner"], survey["norm"]
    halves = _halves(fkp, 1)
    kf = 2 * np.pi / float(np.max(np.asarray(survey["mattrs"].boxsize)))
    kvec, _ = get_planewave_nodes(np.array([0.001, 0.0011, 2.0, 2.2]) * kf, n_k=1, n_directions=4)
    kw = dict(binner=binner, fkp_norms=[norm], batch_size=4, data_is_randoms=True)
    r64 = np.asarray(get_planewave_responses(halves, kvec=kvec, **kw))
    r32 = np.asarray(get_planewave_responses(halves, kvec=kvec, mesh_dtype="float32", **kw))
    nk = r64.shape[-1] // 3
    for node in range(kvec.shape[0]):  # observed monopole and quadrupole, bins 1-4 (bin 0: the k = 0 mode only)
        for sl in (slice(1, 5), slice(nk + 1, nk + 5)):
            a, b = r32[node, 0, 0, sl], r64[node, 0, 0, sl]
            assert np.allclose(a, b, rtol=2e-4, atol=1e-6 * np.abs(r64[node, 0, 0]).max())
