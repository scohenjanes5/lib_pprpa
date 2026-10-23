"""GPU active-space GDF factors (lib_pprpa.gpu_gdf) against the CPU PySCF GDF."""

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
