"""LowRankFFTDF: the exchange of an orbital-tagged density against gpu4pyscf's FFTDF, full,
short- and long-range kernels, with and without the Ewald probe charge; the ACE exchange on
the occupied orbitals; hybrid SCF energies and orbitals."""

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


def _scf(cell, xc, lowrank, ace=False):
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    kmf = gdft.KRKS(cell, kpts=KPTS, xc=xc)
    kmf.exxdiv = None
    kmf.conv_tol = 1e-11
    if lowrank:
        kmf.with_df = LowRankFFTDF(cell, KPTS)
        kmf.with_df.ace = ace
    kmf.kernel()
    return kmf


@pytest.mark.parametrize("omega", (None, 0.7, -0.7))
@pytest.mark.parametrize("exxdiv", (None, "ewald"))
def test_ace_exact_on_occupied(omega, exxdiv):
    """K_ace C_occ = K C_occ and Tr(D K_ace) = Tr(D K) against gpu4pyscf's FFTDF."""
    import cupy as cp
    from gpu4pyscf.pbc.df.fft import FFTDF
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    cell = _cell()
    kmf = _scf(cell, "pbe", lowrank=False)
    dm = kmf.make_rdm1()
    args = dict(hermi=1, kpts=KPTS, with_j=False, omega=omega, exxdiv=exxdiv)
    ref = FFTDF(cell, KPTS).get_jk(dm, **args)[1][0]
    df = LowRankFFTDF(cell, KPTS)
    df.ace = True
    got = df.get_jk(dm, **args)[1][0]
    cocc = cp.asarray(kmf.mo_coeff[0])[:, cp.asarray(kmf.mo_occ[0]) > 0]
    scale = float(abs(ref @ cocc).max())
    assert float(abs(got @ cocc - ref @ cocc).max()) < 1e-10 * scale
    e_ref, e_got = (float(cp.einsum('ij,ji->', cp.asarray(dm[0]), k)) for k in (ref, got))
    assert abs(e_got - e_ref) < 1e-10 * abs(e_ref)


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


def test_exchange_with_padded_mos():
    """gpu4pyscf pads the MOs of an overlap with removed eigenvalues with zero columns: the
    exchange of such a tagged density is finite and exact on the span of the other columns."""
    import cupy as cp
    from gpu4pyscf.lib.cupy_helper import tag_array
    from gpu4pyscf.pbc.df.fft import FFTDF
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    cell = _cell()
    kmf = _scf(cell, "pbe", lowrank=False)
    dm = kmf.make_rdm1()
    mo = cp.array(kmf.mo_coeff)
    mo[0][:, -2:] = 0                        # two virtuals, as for two removed eigenvalues
    args = dict(hermi=1, kpts=KPTS, with_j=False, exxdiv=None)
    got = LowRankFFTDF(cell, KPTS).get_jk(
        tag_array(cp.asarray(dm), mo_coeff=mo, mo_occ=kmf.mo_occ), **args)[1][0]
    assert bool(cp.isfinite(got).all())
    K = FFTDF(cell, KPTS).get_jk(cp.asarray(dm), **args)[1][0]
    S = cp.asarray(cell.pbc_intor("int1e_ovlp", hermi=1))
    C = mo[0][:, :-2]
    P = C @ cp.linalg.solve(C.T @ S @ C, C.T @ S)
    ref = cp.asnumpy(P.T @ K @ P)
    assert np.abs(cp.asnumpy(got) - ref).max() < 1e-10 * np.abs(ref).max()


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


def test_ket_blocks_start_alike(monkeypatch):
    """Every ket block starts with the device memory of the first: the plan is made once, so
    nothing of a block (the last AO chunk of its projection) may stay into the next."""
    import cupy as cp
    import lib_pprpa.gpu_exchange as gx
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    cell = _cell()
    dm = _scf(cell, "pbe", lowrank=False).make_rdm1()
    ngrid = int(np.prod(cell.mesh))
    monkeypatch.setattr(gx, "free_bytes", lambda: 240 * ngrid + 512 * 1024**2)
    pool = cp.get_default_memory_pool()
    used = []
    grids = gx._grids

    def recording(*args):
        used.append(pool.used_bytes())
        return grids(*args)

    monkeypatch.setattr(gx, "_grids", recording)
    LowRankFFTDF(cell, KPTS).get_jk(dm, hermi=1, kpts=KPTS, with_j=False, exxdiv=None)
    assert len(used) > 2
    # only the previous block's (nao, 4) K block may remain (512 bytes); a kept last AO
    # chunk is (nao, 2000) here, 256 kB
    assert max(used) - used[0] < 4096


@pytest.mark.parametrize("xc", ("b3lyp", "hse06"))
def test_hybrid_scf_energy(xc):
    cell = _cell()
    ref, got = (_scf(cell, xc, lowrank) for lowrank in (False, True))
    assert got.converged
    assert abs(got.e_tot - ref.e_tot) < 1e-9


@pytest.mark.parametrize("xc", ("b3lyp", "hse06"))
def test_ace_scf(xc):
    """An ACE SCF converges to the full-exchange energy and density; full_exchange_orbitals
    then gives the full-exchange orbital energies, virtuals included."""
    import cupy as cp
    from lib_pprpa.gpu_fftdf import full_exchange_orbitals
    cell = _cell()
    ref, got = _scf(cell, xc, lowrank=False), _scf(cell, xc, lowrank=True, ace=True)
    assert got.converged
    assert abs(got.e_tot - ref.e_tot) < 1e-9
    assert float(abs(cp.asarray(got.make_rdm1()) - cp.asarray(ref.make_rdm1())).max()) < 1e-6
    # the ACE Fock lacks the virtuals' exchange
    assert float(abs(cp.asarray(got.mo_energy) - cp.asarray(ref.mo_energy)).max()) > 1e-4
    full_exchange_orbitals(got)
    assert got.with_df.ace
    assert float(abs(cp.asarray(got.mo_energy) - cp.asarray(ref.mo_energy)).max()) < 1e-6
    assert float(abs(cp.asarray(got.make_rdm1()) - cp.asarray(ref.make_rdm1())).max()) < 1e-6
