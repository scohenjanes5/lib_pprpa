"""LowRankFFTDF: the exchange of an orbital-tagged density against gpu4pyscf's FFTDF, full,
short- and long-range kernels, with and without the Ewald probe charge; hybrid SCF energies."""

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

KPTS = np.zeros((1, 3))


def _cell():
    from pyscf.pbc import gto
    a0 = 3.370137329
    return gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2 + 0.1, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)


def _scf(cell, xc, lowrank):
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    kmf = gdft.KRKS(cell, kpts=KPTS, xc=xc)
    kmf.exxdiv = None
    kmf.conv_tol = 1e-11
    if lowrank:
        kmf.with_df = LowRankFFTDF(cell, KPTS)
    kmf.kernel()
    return kmf


@pytest.mark.parametrize("omega", (None, 0.7, -0.7))
@pytest.mark.parametrize("exxdiv", (None, "ewald"))
def test_exchange_matches_fftdf(omega, exxdiv):
    import cupy as cp
    from gpu4pyscf.pbc.df.fft import FFTDF
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    cell = _cell()
    kmf = _scf(cell, "pbe", lowrank=False)
    dm = kmf.make_rdm1()
    assert getattr(dm, "mo_coeff", None) is not None
    args = dict(hermi=1, kpts=KPTS, with_j=False, omega=omega, exxdiv=exxdiv)
    ref = cp.asnumpy(FFTDF(cell, KPTS).get_jk(dm, **args)[1])
    got = cp.asnumpy(LowRankFFTDF(cell, KPTS).get_jk(dm, **args)[1])
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-10 * np.abs(ref).max()
    # an untagged density (the SCF's initial guess) is factored over the AO basis
    r = np.random.default_rng(2).standard_normal((cell.nao,) * 2)
    dm_plain = cp.asarray(np.asarray(dm.get())[0] + 0.01 * (r + r.T))[None]
    ref = cp.asnumpy(FFTDF(cell, KPTS).get_jk(dm_plain, **args)[1])
    got = cp.asnumpy(LowRankFFTDF(cell, KPTS).get_jk(dm_plain, **args)[1])
    assert np.abs(got - ref).max() < 1e-10 * np.abs(ref).max()


def test_exchange_in_blocks(monkeypatch):
    """A free-memory figure that forces ket blocks of 4, factor blocks of 12 and single-pair
    FFT batches reproduces the one-block exchange, tagged (rank nocc) and untagged (rank nao)."""
    import cupy as cp
    import lib_pprpa.gpu_exchange as gx
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    cell = _cell()
    kmf = _scf(cell, "pbe", lowrank=False)
    dm = kmf.make_rdm1()
    r = np.random.default_rng(2).standard_normal((cell.nao,) * 2)
    dm_plain = cp.asarray(np.asarray(dm.get())[0] + 0.01 * (r + r.T))[None]
    args = dict(hermi=1, kpts=KPTS, with_j=False, exxdiv=None)
    one = [cp.asnumpy(LowRankFFTDF(cell, KPTS).get_jk(d, **args)[1]) for d in (dm, dm_plain)]
    ngrid = int(np.prod(cell.mesh))
    # plan_budget keeps 512 MiB back: 240 bytes per grid point remain
    monkeypatch.setattr(gx, "free_bytes", lambda: 240 * ngrid + 512 * 1024**2)
    assert gx._plan(cell, cell.mesh, cell.nao, cell.nao) == (4, 12, 1)
    for d, ref in zip((dm, dm_plain), one):
        got = cp.asnumpy(LowRankFFTDF(cell, KPTS).get_jk(d, **args)[1])
        assert np.abs(got - ref).max() < 1e-12 * np.abs(ref).max()


@pytest.mark.parametrize("xc", ("b3lyp", "hse06"))
def test_hybrid_scf_energy(xc):
    cell = _cell()
    ref, got = (_scf(cell, xc, lowrank) for lowrank in (False, True))
    assert got.converged
    assert abs(got.e_tot - ref.e_tot) < 1e-9
