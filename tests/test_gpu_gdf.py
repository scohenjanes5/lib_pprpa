"""GPU active-space GDF factors (lib_pprpa.gpu_gdf) against the CPU PySCF GDF."""

import copy

import numpy as np
import pytest


def _gpu4pyscf_available():
    try:
        import cupy as cp
        import gpu4pyscf  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu4pyscf_available(), reason="GPU4PySCF CUDA device is unavailable")


def _diamond(with_j3c):
    from pyscf.pbc import df, dft, gto

    a = 3.567 / 0.52917721092
    frac = np.array([[0, 0, 0], [.25, .25, .25], [0, .5, .5], [.25, .75, .75],
                     [.5, 0, .5], [.75, .25, .75], [.5, .5, 0], [.75, .75, .25]])
    shift = 0.02 * np.array([[1, -2, 3]]) * (np.arange(8)[:, None] % 3)
    cell = gto.Cell()
    cell.atom = [("C", x) for x in frac * a + shift]
    cell.a = np.eye(3) * a
    cell.unit = "Bohr"
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pbe"
    cell.mesh = [15, 15, 15]
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.with_df = df.GDF(cell, kpts=np.zeros((1, 3)))
    mf.with_df.auxbasis = "weigend"
    mf.with_df.linear_dep_threshold = 1e-10
    mf.with_df.build(j_only=False, with_j3c=with_j3c)
    # orthonormal (Lowdin) orbitals: the factors do not need an SCF
    s_e, s_u = np.linalg.eigh(mf.get_ovlp())
    mf.mo_coeff = s_u / np.sqrt(s_e)[None, :]
    mf.mo_energy = np.arange(cell.nao, dtype=float)
    mf.mo_occ = np.zeros(cell.nao)
    mf.mo_occ[:cell.nelectron // 2] = 2.0
    mf.e_tot = 0.0
    mf.converged = True
    return mf


@pytest.fixture(scope="module")
def diamond():
    return _diamond(with_j3c=True)


def _eri(lpq):
    return np.einsum("Lpq,Lrs->pqrs", lpq, lpq)


def test_gpu_factors_reproduce_gdf_integrals(diamond):
    import cupy as cp
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
    from lib_pprpa.pbc_gdf import get_pyscf_input_sc_gdf

    nocc, moe, lpq_cpu = get_pyscf_input_sc_gdf(diamond, nocc_act=12, nvir_act=10)
    nocc_g, moe_g, lpq_gpu = get_pyscf_input_sc_gdf(diamond, nocc_act=12, nvir_act=10,
                                                     gpu=True)
    assert nocc_g == nocc == 12
    np.testing.assert_array_equal(moe_g, moe)
    assert lpq_gpu.shape == lpq_cpu.shape
    # the same integrals as gpu4pyscf's GDF build (its compressed AO cderi, unpacked)
    C = cp.asarray(diamond.mo_coeff[:, 4:26])
    lpq_stock = cp.concatenate([
        cp.matmul(C.T, cp.matmul(L, C)) for L, _, _ in
        make_gpu_gdf_mf(diamond).with_df.loop_gamma_point(64)]).get()
    np.testing.assert_allclose(_eri(lpq_gpu), _eri(lpq_stock), rtol=0.0, atol=1e-12)
    # gpu4pyscf's and PySCF's GDF builders differ at ~1e-7 here (screening at precision 1e-8)
    np.testing.assert_allclose(_eri(lpq_gpu), _eri(lpq_cpu), rtol=0.0, atol=2e-7)


def test_gpu_factors_independent_of_blocking(diamond):
    from lib_pprpa.gpu_gdf import gdf_active_factors

    mo = diamond.mo_coeff[:, 4:26]
    ref = gdf_active_factors(diamond.with_df, mo)
    # a budget of a few MB splits the aux functions and the G vectors into several blocks
    small = gdf_active_factors(diamond.with_df, mo, max_memory=5_000_000)
    np.testing.assert_allclose(_eri(small), _eri(ref), rtol=0.0, atol=1e-12)


def test_gpu_factors_need_no_ao_cderi():
    from lib_pprpa.pbc_gdf import make_pprpa_gdf

    mf = _diamond(with_j3c=False)
    solver = make_pprpa_gdf(mf, channel="pp", nocc_act=8, nvir_act=8, nroot=1, gpu=True)
    assert mf.with_df._cderi is None
    assert solver.Lpq.shape[1:] == (16, 16)


def test_cderi_factors_match_streamed(diamond):
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
    from lib_pprpa.gpu_gdf import cderi_active_factors, gdf_active_factors

    gpu_df = make_gpu_gdf_mf(diamond).with_df
    mo = diamond.mo_coeff[:, 4:26]
    ref = gdf_active_factors(diamond.with_df, mo)
    lpq = cderi_active_factors(gpu_df, mo)
    assert lpq.shape == ref.shape and lpq.flags.c_contiguous
    np.testing.assert_allclose(_eri(lpq), _eri(ref), rtol=0.0, atol=1e-12)
    # a budget of a few MB splits the cderi rows into several blocks
    small = cderi_active_factors(gpu_df, mo, max_memory=2_000_000)
    np.testing.assert_allclose(small, lpq, rtol=0.0, atol=1e-14)


def test_make_pprpa_gdf_reads_gpu_df(diamond):
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
    from lib_pprpa.pbc_gdf import make_pprpa_gdf, validate_gdf_context

    gpu_df = make_gpu_gdf_mf(diamond).with_df
    solver = make_pprpa_gdf(diamond, channel="pp", nocc_act=8, nvir_act=8, nroot=1,
                            gpu_df=gpu_df)
    streamed = make_pprpa_gdf(diamond, channel="pp", nocc_act=8, nvir_act=8, nroot=1,
                              gpu=True)
    np.testing.assert_allclose(_eri(solver.Lpq), _eri(streamed.Lpq), rtol=0.0, atol=1e-12)
    validate_gdf_context(solver, diamond)


def _tiered_twin(diamond, tmp_path, ndev, nhost):
    """The stock GPU GDF of ``diamond`` and a tiered twin with ndev device rows, nhost host rows
    and the rest in a scratch file under tmp_path."""
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
    from lib_pprpa.gpu_gdf import build_tiered_gdf

    stock = make_gpu_gdf_mf(diamond).with_df
    npair = stock._cderi[0].shape[1]
    twin = copy.copy(stock)
    twin.reset()
    build_tiered_gdf(twin, device_bytes=ndev * 8 * npair, host_bytes=nhost * 8 * npair,
                     scratch_dir=str(tmp_path))
    return stock, twin


def test_tiered_cderi_matches_stock_build(diamond, tmp_path):
    from lib_pprpa.gpu_gdf import cderi_active_factors

    stock, twin = _tiered_twin(diamond, tmp_path, ndev=40, nhost=70)
    store = twin._cderi[0]
    naux = stock._cderi[0].shape[0]
    assert store.shape == stock._cderi[0].shape
    assert (store.ndev, store.nhost) == (40, 70) and store.nfile == naux - 110
    np.testing.assert_array_equal(twin._cderi_idx[0].get(), stock._cderi_idx[0].get())
    full = store[0:naux].get()
    np.testing.assert_allclose(full, stock._cderi[0], rtol=0.0, atol=1e-12)
    # slices inside one tier and across the device/host and host/file boundaries
    for q0, q1 in ((3, 17), (30, 50), (100, 120), (35, naux - 2), (115, naux)):
        np.testing.assert_array_equal(store[q0:q1].get(), full[q0:q1])
    mo = diamond.mo_coeff[:, 4:26]
    np.testing.assert_allclose(_eri(cderi_active_factors(twin, mo)),
                               _eri(cderi_active_factors(stock, mo)), rtol=0.0, atol=1e-12)


def test_tiered_cderi_through_gpu4pyscf_jk(diamond, tmp_path):
    import cupy as cp

    stock, twin = _tiered_twin(diamond, tmp_path, ndev=40, nhost=70)
    rng = np.random.default_rng(7)
    dm = rng.standard_normal((diamond.cell.nao,) * 2)
    dm = cp.asarray(dm + dm.T)
    vj, vk = stock.get_jk(dm, hermi=1, exxdiv=None)
    tj, tk = twin.get_jk(dm, hermi=1, exxdiv=None)
    np.testing.assert_allclose(tj.get(), vj.get(), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(tk.get(), vk.get(), rtol=0.0, atol=1e-10)


def test_cderi_pair_factors_layout(diamond, tmp_path):
    from lib_pprpa.gpu_gdf import cderi_active_factors, cderi_pair_factors
    from lib_pprpa.pprpa_eri_gpu import gdf_pair_factors

    _, twin = _tiered_twin(diamond, tmp_path, ndev=40, nhost=70)
    nocc = 12
    mo = diamond.mo_coeff[:, 4:26]
    ref = gdf_pair_factors(cderi_active_factors(twin, mo), nocc)
    F = cderi_pair_factors(twin, mo, nocc)
    assert set(F) == set(ref)
    for key in ref:
        assert F[key].flags.c_contiguous and F[key].shape == ref[key].shape
        np.testing.assert_allclose(F[key].get(), ref[key].get(), rtol=0.0, atol=1e-14)


def test_operator_from_cderi_factors_roots(diamond, tmp_path):
    from lib_pprpa.gpu_gdf import cderi_active_factors, cderi_pair_factors
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import (attach_gpu_eri_contraction, gdf_pair_operator,
                                         pair_operator_from_factors, release_gpu_eri)

    _, twin = _tiered_twin(diamond, tmp_path, ndev=40, nhost=70)
    nocc, mo = 8, diamond.mo_coeff[:, 8:24]
    moe = np.linspace(-1.0, 1.0, 16)
    lpq = cderi_active_factors(twin, mo)
    exci = {}
    for name, op in (("lpq", gdf_pair_operator(lpq, nocc, "t")),
                     ("cderi", pair_operator_from_factors(cderi_pair_factors(twin, mo, nocc),
                                                          nocc, "t"))):
        mp = ppRPA_Davidson(nocc, moe, None, channel="pp", nroot=3, residue_thresh=1e-10)
        attach_gpu_eri_contraction(mp, op)
        mp.kernel("t")
        exci[name] = np.asarray(mp.exci_t)
        release_gpu_eri(mp)
    np.testing.assert_allclose(exci["cderi"], exci["lpq"], rtol=0.0, atol=1e-10)


def test_make_pprpa_gdf_operator_route(diamond, tmp_path):
    import pytest
    from lib_pprpa.pbc_gdf import make_pprpa_gdf, validate_gdf_context
    from lib_pprpa.pprpa_eri_gpu import release_gpu_eri

    _, twin = _tiered_twin(diamond, tmp_path, ndev=40, nhost=70)
    kw = dict(channel="pp", nocc_act=8, nvir_act=8, nroot=3, residue_thresh=1e-10)
    ref = make_pprpa_gdf(diamond, gpu_df=twin, **kw)
    ref.kernel("t")
    solver = make_pprpa_gdf(diamond, gpu_df=twin, operator="t", **kw)
    assert solver.Lpq is None
    solver.kernel("t")
    np.testing.assert_allclose(solver.exci_t, ref.exci_t, rtol=0.0, atol=1e-10)
    validate_gdf_context(solver, diamond)
    # the gradient never uses the operator, so releasing it keeps the provenance
    release_gpu_eri(solver)
    validate_gdf_context(solver, diamond)
    with pytest.raises(ValueError):
        make_pprpa_gdf(diamond, operator="t", **kw)


def test_tiered_host_share_follows_max_memory(diamond, tmp_path):
    from pyscf import lib
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import make_gpu_gdf_mf
    from lib_pprpa.gpu_gdf import build_tiered_gdf

    stock = make_gpu_gdf_mf(diamond).with_df
    naux = stock._cderi[0].shape[0]
    twin = copy.copy(stock)
    used = lib.current_memory()[0]
    # no room beside what the process holds: every row not on the device goes to scratch
    twin.reset()
    build_tiered_gdf(twin, device_bytes=0, max_memory=used / 2, scratch_dir=str(tmp_path))
    store = twin._cderi[0]
    assert (store.ndev, store.nhost, store.nfile) == (0, 0, naux)
    np.testing.assert_allclose(store[0:naux].get(), stock._cderi[0], rtol=0.0, atol=1e-12)
    # plenty of room: all of it in host memory
    twin.reset()
    build_tiered_gdf(twin, max_memory=used + 1e6)
    assert (twin._cderi[0].nhost, twin._cderi[0].nfile) == (naux, 0)
