"""One-pass GDF J/K derivatives of several densities against gpu4pyscf's one call per density."""

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


@pytest.fixture(scope="module")
def engine():
    from pyscf.pbc import df, dft, gto
    from lib_pprpa.grad.pprpa_gamma_gdf_gpu import _GDFDerivativeEngine, make_gpu_gdf_mf

    # a small cubic box: a skewed primitive cell needs many more lattice images
    cell = gto.Cell()
    cell.atom = [("H", (0.03, -0.02, 0.01)), ("H", (1.45, 0.2, -0.1))]
    cell.a = np.diag([5.0, 5.2, 5.4])
    cell.unit = "Bohr"
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pbe"
    cell.ke_cutoff = 40.0
    cell.precision = 1e-8
    cell.verbose = 0
    cell.build()

    gamma = np.zeros((1, 3))
    mf = dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.with_df = df.GDF(cell, kpts=gamma)
    mf.with_df.auxbasis = "weigend"
    mf.with_df.linear_dep_threshold = 1e-10
    mf.with_df.build(j_only=False, with_j3c=False)
    s_e, s_u = np.linalg.eigh(mf.get_ovlp())
    mf.mo_coeff = s_u / np.sqrt(s_e)[None, :]
    mf.mo_energy = np.arange(cell.nao, dtype=float)
    mf.mo_occ = np.zeros(cell.nao)
    mf.mo_occ[:cell.nelectron // 2] = 2.0
    mf.e_tot = 0.0
    mf.converged = True
    return _GDFDerivativeEngine(make_gpu_gdf_mf(mf))


def _separate(engine, dm, hermi, j_factor, k_factor, exxdiv):
    """gpu4pyscf's Gamma-point derivative of one density."""
    from gpu4pyscf.pbc.df.grad.rhf import _jk_energy_per_atom
    return np.asarray(_jk_energy_per_atom(
        engine.opt, dm, hermi=hermi, j_factor=j_factor, k_factor=k_factor, exxdiv=exxdiv,
        omega=0.0, linear_dep_threshold=engine.linear_dep_threshold))


def _densities(nao):
    import cupy as cp
    rng = np.random.default_rng(23)
    c = rng.standard_normal((nao, nao))
    d = c[:, :3] @ c[:, :3].T * 0.3              # PSD, like a reference density
    p = rng.standard_normal((nao, nao))
    p = (p + p.T) * 0.05                         # indefinite, like a relaxed correction
    x = rng.standard_normal((nao, nao)) * 0.1
    return [cp.asarray(a) for a in (d + p, p, (x - x.T) * 0.5, (x + x.T) * 0.5)]


@pytest.mark.parametrize("hybrid, exxdiv", [(0.0, None), (0.25, None), (0.25, "ewald")])
def test_one_pass_matches_separate_calls(engine, hybrid, exxdiv):
    """The pp-RPA terms Q(D+P) - Q(P) + K(antisymmetric X) - K(symmetric X) in one pass."""
    t, p, xa, xs = _densities(engine.gpu_mf.cell.nao)
    terms = [(t, 1, 1.0, hybrid, exxdiv, 1.0), (p, 1, 1.0, hybrid, exxdiv, -1.0),
             (xa, 2, 0.0, 2.0, None, 1.0), (xs, 1, 0.0, 2.0, None, -1.0)]
    ref = sum(w * _separate(engine, dm, h, j, k, e) for dm, h, j, k, e, w in terms)
    de = engine.jk_sum(terms)
    assert de.shape == ref.shape == (engine.gpu_mf.cell.natm, 3)
    np.testing.assert_allclose(de, ref, rtol=0, atol=1e-9 * max(1.0, abs(ref).max()))


def test_gamma_antisymmetric_matches_kpoint_path(engine):
    """An antisymmetric density takes the Gamma path, which pbc.df.grad.krhf sends through its
    complex k-point path."""
    from gpu4pyscf.pbc.df.grad.krhf import _jk_energy_per_atom
    xa = _densities(engine.gpu_mf.cell.nao)[2]
    de = engine.jk_sum([(xa, 2, 0.0, 2.0, None, 1.0)])
    ref = np.asarray(_jk_energy_per_atom(
        engine.opt, xa[None], engine.kpts, hermi=2, j_factor=0.0, k_factor=2.0,
        exxdiv=None, omega=0.0, linear_dep_threshold=engine.linear_dep_threshold))
    np.testing.assert_allclose(de, ref, rtol=0, atol=1e-9 * max(1.0, abs(ref).max()))
