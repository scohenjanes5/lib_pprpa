"""gpu_fft_chain against codensity + cuFFT + the one-pass accumulation, and the exchange of
gpu_exchange on both routes."""

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


def test_applies():
    from lib_pprpa.gpu_fft_chain import applies
    assert applies((20, 18, 27)) and applies((21, 25, 16)) and applies((160, 160, 160))
    assert not applies((107, 107, 107)) and not applies((20, 22, 20))


@pytest.mark.parametrize("mesh", [(24, 20, 30), (20, 18, 27), (21, 25, 16), (40, 36, 45),
                                  (48, 32, 64), (9, 7, 10)])
def test_chain_against_cufft(mesh):
    """Repeated, unsorted target rows, a row without contributions, a batch that starts at p0 > 0;
    any w(G) once half_kernel has symmetrised it."""
    import cupy as cp
    import lib_pprpa.gpu_exchange as gx
    from lib_pprpa.gpu_coulomb import codensity, coulomb_potential, half_kernel
    from lib_pprpa.gpu_fft_chain import FFTChain
    rng = np.random.default_rng(5)
    ngrid = int(np.prod(mesh))
    A, B, S = (cp.asarray(rng.standard_normal((n, ngrid))) for n in (4, 3, 4))
    w_half = half_kernel(cp.asarray(rng.random(ngrid)), mesh)
    aidx = cp.asarray([0, 1, 2, 3, 0, 1, 2, 3, 1], dtype=np.int32)
    bidx = cp.asarray([0, 0, 1, 2, 2, 1, 0, 2, 1], dtype=np.int32)
    p0, p1 = 2, 9
    target = np.array([3, 0, 3, 1, 3, 0, 4])
    pair = np.arange(p1 - p0)
    src = np.array([1, 2, 0, 3, 1, 2, 0])
    scale = cp.asarray(rng.standard_normal(4))
    W0 = cp.asarray(rng.standard_normal((6, ngrid)))
    ref = W0.copy()
    V = coulomb_potential(codensity(A, B, aidx, bidx, p0, p1), mesh, w_half)
    gx._accumulate(ref, V, S, scale, target, pair, src)
    got = W0.copy()
    FFTChain(mesh, w_half, p1 - p0).add(got, A, B, aidx, bidx, p0, p1, S, scale,
                                        *gx._csr(target, pair, src))
    err = float(cp.abs(got - ref).max() / cp.abs(ref - W0).max())
    assert err < 1e-13
    assert bool((got[[2, 5]] == W0[[2, 5]]).all())


def test_exchange_routes_agree(monkeypatch):
    """exchange_ket and exchange_cross_ket on a 7-smooth mesh: the five kernels against cuFFT."""
    import cupy as cp
    from pyscf.pbc import gto
    from gpu4pyscf.pbc import tools as gtools
    import lib_pprpa.gpu_exchange as gx
    import lib_pprpa.gpu_fft_chain as fc
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]), unit="bohr",
                 basis="gth-dzv", pseudo="gth-pade", mesh=[18, 20, 21], verbose=0)
    assert fc.applies(cell.mesh)
    coulG = gtools.get_coulG(cell, mesh=cell.mesh)
    rng = np.random.default_rng(7)
    U, ket, cross = (rng.standard_normal((cell.nao, n)) for n in (6, 4, 4))
    lam = rng.standard_normal(6)

    def both():
        return (cp.asnumpy(gx.exchange_ket(cell, cell.mesh, lam, U, ket, coulG)),
                cp.asnumpy(gx.exchange_cross_ket(cell, cell.mesh, ket, cross, coulG)))

    got = both()
    monkeypatch.setattr(fc, "applies", lambda mesh: False)
    for g, r in zip(got, both()):
        assert np.abs(g - r).max() < 1e-12 * np.abs(r).max()
