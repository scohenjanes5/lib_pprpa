"""The fused Gamma response (J inside the fxc pass, a hybrid's exchange from the MO factors)
against gpu4pyscf's KRKS.gen_response: the full matrix and the columns of the occupied ket,
for a CPHF (vir-occ) and a general density, LDA, GGA and hybrids; make_gpu_vresp's routing."""

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


def _diamond(xc):
    from pyscf.pbc import dft, gto
    a0 = 3.370137329
    cell = gto.M(atom=[["C", [0., 0., 0.]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
                 a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
                 unit="bohr", basis="gth-dzv", pseudo="gth-pade", mesh=[20] * 3, verbose=0)
    mf = dft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return cell, mf


def _trial(mf):
    occ = mf.mo_occ
    o, v = mf.mo_coeff[:, occ > 0], mf.mo_coeff[:, occ == 0]
    t = np.random.default_rng(0).standard_normal((v.shape[1], o.shape[1]))
    dm = v @ t @ o.T
    return o, v, t, dm + dm.T


@pytest.mark.parametrize("xc", ("pbe", "lda,vwn", "pbe0", "b3lyp",
                                "0.25*SR_HF(1.0) + 0.75*PBE, PBE",
                                "0.4*LR_HF(1.0) + 0.6*PBE, PBE"))
def test_fused_response_matches_gen_response(xc):
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_response import fused_response
    cell, mf = _diamond(xc)
    o, v, t, dm = _trial(mf)
    r = np.random.default_rng(1).standard_normal((cell.nao,) * 2)
    dm_general = mf.mo_coeff @ (r + r.T) @ mf.mo_coeff.T
    kmf = gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc)
    kmf.exxdiv = None
    kmf.mo_coeff = cp.asarray(mf.mo_coeff)[None]
    kmf.mo_energy = cp.asarray(mf.mo_energy)[None]
    kmf.mo_occ = cp.asarray(mf.mo_occ)[None]
    kmf.grids.build()
    nao = cell.nao
    native = kmf.gen_response(singlet=None, hermi=1)
    respond = fused_response(cell, kmf)
    # the exchange's FFTs run real-to-complex: 1e-9 for hybrids
    tol = 1e-9 if mf._numint.libxc.is_hybrid_xc(xc) else 1e-10
    for d in (dm, dm_general):
        ref = cp.asnumpy(native(cp.asarray(d).reshape(1, 1, nao, nao))[0, 0])
        got = cp.asnumpy(respond(cp.asarray(d)))
        assert np.abs(got - ref).max() < tol * np.abs(ref).max()
        got_ket = cp.asnumpy(respond(cp.asarray(d), ket=cp.asarray(o)))
        assert np.abs(got_ket - ref @ o).max() < tol * np.abs(ref @ o).max()
    # the CPHF density stated as cross o^T + o cross^T: its o o pairs taken once
    ref = cp.asnumpy(native(cp.asarray(dm).reshape(1, 1, nao, nao))[0, 0]) @ o
    got = cp.asnumpy(respond(cp.asarray(dm), ket=cp.asarray(o), cross=cp.asarray(v @ t)))
    assert np.abs(got - ref).max() < tol * np.abs(ref).max()


def test_make_gpu_vresp_routes():
    """LDA/GGA and their hybrids on FFTDF get the fused response, GDF gen_response; the
    response serves ``ket``."""
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.grad.pprpa_gamma_gpu import _fused_applies, make_gpu_vresp
    cell, mf = _diamond("pbe")
    o, v, t, dm = _trial(mf)
    assert not _fused_applies(gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc="pbe").density_fit())
    # the ket and the full matrix factor a hybrid's density differently: 1e-9
    for xc, tol in (("pbe", 1e-10), ("pbe0", 1e-9)):
        mf.xc = xc
        assert _fused_applies(gdft.KRKS(cell, kpts=np.zeros((1, 3)), xc=xc))
        vresp = make_gpu_vresp(cell, mf, emulate=False)
        ref = vresp(dm) @ o
        np.testing.assert_allclose(vresp(dm, ket=o), ref, rtol=0, atol=tol * np.abs(ref).max())


def test_cross_exchange_in_blocks(monkeypatch):
    """exchange_cross_ket (the o o pairs of a CPHF density once) against exchange_ket of the
    density's factors, in one FFT batch and in batches of two pairs that split the rows."""
    import cupy as cp
    from gpu4pyscf.pbc import tools as gtools
    import lib_pprpa.gpu_exchange as gx
    cell, mf = _diamond("pbe")
    o, v, t, dm = _trial(mf)
    coulG = gtools.get_coulG(cell, mesh=cell.mesh)
    C = np.hstack([o, v])
    proj = np.linalg.pinv(C)
    lam, U = gx.factor(proj @ dm @ proj.T, C)
    ref = cp.asnumpy(gx.exchange_ket(cell, cell.mesh, lam, U, o, coulG))
    got = cp.asnumpy(gx.exchange_cross_ket(cell, cell.mesh, o, v @ t, coulG))
    assert np.abs(got - ref).max() < 1e-11 * np.abs(ref).max()
    ngrid = int(np.prod(cell.mesh))

    def tight():
        # plan_budget keeps 512 MiB back: 160 bytes per grid point beside the held grids
        return 160 * ngrid + 512 * 1024**2

    monkeypatch.setattr(gx, "free_bytes", tight)
    calls = []
    add = gx._PotentialSum.add

    def counted(self, W, A, B, aidx, bidx, p0, p1, *rest):
        calls.append(p1 - p0)
        add(self, W, A, B, aidx, bidx, p0, p1, *rest)

    monkeypatch.setattr(gx._PotentialSum, "add", counted)
    got = cp.asnumpy(gx.exchange_cross_ket(cell, cell.mesh, o, v @ t, coulG))
    assert max(calls) == 2
    assert np.abs(got - ref).max() < 1e-11 * np.abs(ref).max()


def _blocks(nmo, nocc, kind, rng):
    """A symmetric M over the MOs: every block random ('full'), the occupied block, the
    vir-occ block and a rank-1 virtual block ('vo'), the occupations ('occ'), a rank-1
    occupied block ('oo1')."""
    M = np.zeros((nmo, nmo))
    o, v = slice(0, nocc), slice(nocc, nmo)
    if kind == "full":
        M = rng.standard_normal((nmo, nmo))
    elif kind == "vo":
        M[o, o] = rng.standard_normal((nocc, nocc))
        M[v, o] = rng.standard_normal((nmo - nocc, nocc))
        u = rng.standard_normal(nmo - nocc)
        M[v, v] = np.outer(u, u)
    elif kind == "occ":
        M[o, o] = 2 * np.eye(nocc)
    else:
        u = rng.standard_normal(nocc)
        M[o, o] = np.outer(u, u)
    return M + M.T


# the route each density takes on diamond's 4 occupied of 16 MOs: factors 16 x 4 = 64 pairs
# against 10 + 16 + 12 x 4; rank 9 x 4 against 10 + 16 + 4; 4 x 4 against 10; 1 x 4 against 10
_ROUTES = {"full": "factors", "vo": "blocks", "occ": "blocks", "oo1": "factors"}


@pytest.mark.parametrize("kind", sorted(_ROUTES))
def test_occupied_blocks_match_factors(kind, monkeypatch):
    """exchange_occ_ket, K and force (D's occupied/virtual blocks or its factors,
    whichever has fewer pairs) against exchange_ket of D's factors; the
    blocks also in batches of two pairs."""
    import cupy as cp
    from gpu4pyscf.pbc import tools as gtools
    import lib_pprpa.gpu_exchange as gx
    cell, mf = _diamond("pbe")
    mo, occ = mf.mo_coeff, mf.mo_occ > 0
    o = mo[:, occ]
    rng = np.random.default_rng(4)
    M = _blocks(mo.shape[1], o.shape[1], kind, rng)
    parts, _ = gx._occ_parts(M, mo, occ)
    assert ("oo" in parts) == (_ROUTES[kind] == "blocks")
    coulG = gtools.get_coulG(cell, mesh=cell.mesh)
    lam, U = gx.factor(M, mo)
    L = rng.standard_normal(o.shape)
    ref = cp.asnumpy(gx.exchange_ket(cell, cell.mesh, lam, U, o, coulG))
    ref_f = gx.exchange_ket(cell, cell.mesh, lam, U, o, coulG, L=L)
    ngrid = int(np.prod(cell.mesh))
    for budget in (None, 160):
        if budget is not None:
            monkeypatch.setattr(gx, "free_bytes", lambda: budget * ngrid + 512 * 1024**2)
        got = cp.asnumpy(gx.exchange_occ_ket(cell, cell.mesh, M, mo, occ, coulG))
        assert np.abs(got - ref).max() < 1e-11 * np.abs(ref).max()
        got_f = gx.exchange_occ_ket(cell, cell.mesh, M, mo, occ, coulG, L=L)
        assert np.abs(got_f - ref_f).max() < 1e-11 * np.abs(ref_f).max()


def test_accumulate_by_target():
    """The one-pass W update against a loop over its contributions: repeated, unsorted
    targets, a row with none."""
    import cupy as cp
    import lib_pprpa.gpu_exchange as gx
    rng = np.random.default_rng(3)
    V, S = rng.standard_normal((6, 50)), rng.standard_normal((4, 50))
    scale, W = rng.standard_normal(4), rng.standard_normal((5, 50))
    target = np.array([3, 0, 3, 1, 3, 0, 4])
    pair = np.array([0, 1, 2, 3, 4, 5, 0])
    src = np.array([1, 2, 0, 3, 1, 2, 0])
    ref = W.copy()
    for t, p, s in zip(target, pair, src):
        ref[t] += scale[s] * S[s] * V[p]
    got = cp.asarray(W)
    gx._accumulate(got, cp.asarray(V), cp.asarray(S), cp.asarray(scale), target, pair, src)
    assert np.abs(cp.asnumpy(got) - ref).max() < 1e-13
