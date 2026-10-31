"""Gamma-point GPU pp-RPA gradient: the low-rank pairing and reference exchange forces
against gpu4pyscf's AFT exchange derivative and the FFTDF exchange energy, and the full
gradient against the CPU reference, with the CPHF warm start."""

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

XC = "pbe"
CHANNEL = "pp"
KPTS = np.zeros((1, 3))


def _cell():
    """Distorted C2 diamond cell, charge +2."""
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.charge = 2
    cell.ke_cutoff = 100.0
    cell.verbose = 0
    cell.build()
    return cell


@pytest.fixture(scope="module")
def system():
    """The C2 cell with a GPU SCF and its MO ERIs through the GPU ao2mo."""
    import cupy as cp
    from pyscf.pbc import dft as cdft
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    cell = _cell()
    kg = gdft.KRKS(cell, kpts=KPTS, xc=XC)
    kg.exxdiv = None
    kg.conv_tol = 1e-10
    kg.kernel()
    mf = cdft.RKS(cell, xc=XC)
    mf.exxdiv = None
    mf.mo_coeff = cp.asnumpy(kg.mo_coeff[0])
    mf.mo_energy = cp.asnumpy(kg.mo_energy[0])
    mf.mo_occ = cp.asnumpy(kg.mo_occ[0])
    mf.e_tot = float(kg.e_tot)
    mf.converged = True
    nocc = cell.nelectron // 2
    eri = gpu_ao2mo_blocks(cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], cell.mesh)
    return cell, mf, nocc, eri


def _solve(system, mult):
    from lib_pprpa.pprpa_davidson import ppRPA_Davidson
    from lib_pprpa.pprpa_eri_gpu import attach_gpu_eri_contraction
    cell, mf, nocc, eri = system
    mp = ppRPA_Davidson(nocc, mf.mo_energy, Lpq=None, channel=CHANNEL, nroot=2,
                        residue_thresh=1e-10, trial="identity")
    mp.mu = 0.0
    attach_gpu_eri_contraction(mp, *eri, mode="resident")
    mp.kernel(mult)
    xy = np.array((mp.xy_s if mult == "s" else mp.xy_t)[0], copy=True)
    return mp, xy


def _dense_get_k(cell):
    import cupy as cp
    from gpu4pyscf.pbc.df import fft_jk
    from gpu4pyscf.pbc.df.fft import FFTDF
    fdf = FFTDF(cell, KPTS)

    def get_k(dm=None, hermi=1, **kw):
        dmg = cp.asarray(dm)
        single = dmg.ndim == 2
        k = cp.asnumpy(fft_jk.get_k(fdf, dmg[None] if single else dmg, hermi=0,
                                    kpt=np.zeros(3), exxdiv=None))
        return k[0] if single else k
    return get_k


def _aft_exchange(cell, mf, dm, sr_factor, lr_factor, omega=0.0):
    """gpu4pyscf's AFT exchange energy derivative Q_K(dm), NumPy (natm, 3)."""
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from gpu4pyscf.pbc.df.aft import AFTDF
    from gpu4pyscf.pbc.grad import krhf as krhf_g
    kf = gdft.KRKS(cell, kpts=KPTS, xc=mf.xc)
    kf.exxdiv = None
    kf.mo_coeff = [cp.asarray(mf.mo_coeff)]
    kf.mo_energy = [cp.asarray(mf.mo_energy)]
    kf.mo_occ = [cp.asarray(mf.mo_occ)]
    kf.with_df = AFTDF(cell, KPTS)
    kf.with_df.mesh = cell.mesh
    kf.rsjk = None
    return cp.asnumpy(krhf_g.jk_energy_per_atom(
        kf, cp.asarray(dm)[None], KPTS, j_factor=0.0, sr_factor=sr_factor,
        lr_factor=lr_factor, omega=omega, exxdiv=None))


def _aft_pairing_force(cell, mf, X):
    """The AFT kernel the low-rank force replaces: +Q_K(antisym X) - Q_K(sym X), sr=lr=2."""
    Xa, Xs = (X - X.T) * 0.5, (X + X.T) * 0.5
    de = np.zeros((cell.natm, 3))
    if np.abs(Xa).max() > 1e-10:
        de += _aft_exchange(cell, mf, Xa, 2.0, 2.0)
    if np.abs(Xs).max() > 1e-10:
        de -= _aft_exchange(cell, mf, Xs, 2.0, 2.0)
    return de


@pytest.mark.parametrize("mult", ("s", "t"))
@pytest.mark.parametrize("store_phil", (True, False))
@pytest.mark.parametrize("pairs_per_batch", (None, 7))
def test_pairing_force_matches_aft(system, mult, store_phil, pairs_per_batch, monkeypatch):
    """AFT and FFT exchange derivatives agree to ~1e-8; the low-rank form is exact in X.

    phiL stored or rebuilt per block, the pairs in one FFT batch or in batches of 7
    (rows split across batches), all reproduce the stored single-batch result.
    """
    from scipy.linalg import block_diag
    import lib_pprpa.gpu_pairing_force as pf
    from lib_pprpa.grad.grad_utils import get_xy_full
    cell, mf, nocc, _ = system
    mp, xy = _solve(system, mult)
    nvir = cell.nao - nocc
    occ_y, vir_x = get_xy_full(xy, mp.oo_dim, mult, nocc=nocc, nvir=nvir)
    cocc, cvir = mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:]
    X = cvir @ vir_x @ cvir.T + cocc @ occ_y @ cocc.T
    C, M = np.hstack([cvir, cocc]), block_diag(vir_x, occ_y)
    ref = _aft_pairing_force(cell, mf, X)
    one = pf.pairing_k_force_lowrank(cell, cell.mesh, C, M, store_phil=True)
    if pairs_per_batch is not None:
        ngrid = int(np.prod(cell.mesh))
        # plan_budget keeps 512 MB back; the rebuilt rows take one more row
        budget = ((pairs_per_batch + 0.5) * (pf._FFT_BYTES + 8) + 8) * ngrid + 512 * 1024**2
        monkeypatch.setattr(pf, "free_bytes", lambda: int(budget))
    got = pf.pairing_k_force_lowrank(cell, cell.mesh, C, M, store_phil=store_phil)
    assert np.abs(got - ref).max() < 1e-6 * np.abs(ref).max()
    assert np.abs(got - one).max() < 1e-11 * np.abs(one).max()


def _exchange_force(cell, mo, M, coulG):
    """F(M) = F_b(M, M) of mo M mo^T through gpu_exchange (W and L from the same density)."""
    from lib_pprpa.gpu_exchange import exchange_force, factor
    lam, U = factor(M, mo)
    return exchange_force(cell, cell.mesh, lam, U, mo, mo @ M, coulG)


# (omega, alpha, hyb): short-range only, long-range only, global
@pytest.mark.parametrize("omega, alpha, hyb",
                         ((1.0, 0.0, 0.25), (1.0, 0.4, 0.0), (0.0, 0.25, 0.25)))
def test_reference_exchange_force_matches_aft(system, omega, alpha, hyb):
    """-1/2 the low-rank force with the hybrid's kernel is gpu4pyscf's AFT exchange
    derivative jk_energy_per_atom(sr=hyb, lr=alpha) of the symmetric density mo M mo^T."""
    from lib_pprpa.gpu_coulomb import exchange_coulG
    cell, mf, _, _ = system
    mo = mf.mo_coeff
    M = 0.1 * np.random.default_rng(3).standard_normal((mo.shape[1],) * 2)
    M = np.diag(mf.mo_occ) + M + M.T
    coulG = exchange_coulG(cell, cell.mesh, omega, alpha, hyb, None)
    got = -0.5 * _exchange_force(cell, mo, M, coulG)
    ref = _aft_exchange(cell, mf, mo @ M @ mo.T, hyb, alpha, omega)
    assert np.abs(got - ref).max() < 1e-6 * np.abs(ref).max()


def test_reference_exchange_bilinear_split(system):
    """F(T) - F(P) = F_b(D, T) + F_b(P, D) for D = diag(mo_occ), as the gradient forms it
    (W of D on every MO, W of P on the occupied ones)."""
    from lib_pprpa.gpu_coulomb import exchange_coulG
    from lib_pprpa.gpu_exchange import exchange_force, factor
    cell, mf, _, _ = system
    mo = mf.mo_coeff
    P = 0.05 * np.random.default_rng(5).standard_normal((mo.shape[1],) * 2)
    P = P + P.T
    D = np.diag(mf.mo_occ)
    coulG = exchange_coulG(cell, cell.mesh, 0.0, 0.25, 0.25, None)
    ref = _exchange_force(cell, mo, D + P, coulG) - _exchange_force(cell, mo, P, coulG)
    occ = mf.mo_occ > 0
    n, cocc = mf.mo_occ[occ], mo[:, occ]
    lam_p, U_p = factor(P, mo)
    got = exchange_force(cell, cell.mesh, n, cocc, mo, mo @ (D + P), coulG)
    got += exchange_force(cell, cell.mesh, lam_p, U_p, cocc, cocc * n, coulG)
    assert np.abs(got - ref).max() < 1e-10 * np.abs(ref).max()


def test_reference_exchange_force_is_fftdf_derivative(system):
    """The low-rank force differentiates the SCF's FFTDF exchange energy -1/4 tr(D K[D]) of
    a fixed AO density on the fixed mesh (central differences, 1e-4 bohr)."""
    import cupy as cp
    from pyscf.pbc import gto
    from gpu4pyscf.pbc.df import fft_jk
    from gpu4pyscf.pbc.df.fft import FFTDF
    from lib_pprpa.gpu_coulomb import exchange_coulG
    cell, mf, _, _ = system
    D = mf.make_rdm1()
    coulG = exchange_coulG(cell, cell.mesh, 0.0, 1.0, 1.0, None)
    got = -0.5 * _exchange_force(cell, mf.mo_coeff, np.diag(mf.mo_occ), coulG)

    def energy(coords):
        c = gto.Cell(atom=[[cell.atom_symbol(i), r] for i, r in enumerate(coords)],
                     a=cell.lattice_vectors(), unit="Bohr", basis=cell.basis,
                     pseudo=cell.pseudo, charge=cell.charge, mesh=cell.mesh, verbose=0)
        c.build()
        k = fft_jk.get_k(FFTDF(c, KPTS), cp.asarray(D)[None], hermi=1, kpt=np.zeros(3),
                         exxdiv=None)
        return -0.25 * float(cp.sum(k[0] * cp.asarray(D)))

    h = 1e-4
    x0 = cell.atom_coords()
    fd = np.zeros_like(got)
    for ia in range(cell.natm):
        for x in range(3):
            xp, xm = x0.copy(), x0.copy()
            xp[ia, x] += h
            xm[ia, x] -= h
            fd[ia, x] = (energy(xp) - energy(xm)) / (2 * h)
    assert np.abs(got - fd).max() < 1e-6 * np.abs(fd).max()


@pytest.mark.parametrize("mult", ("s", "t"))
def test_grad_elec_matches_cpu_and_warm_start(system, mult):
    from lib_pprpa.grad import pprpa_gamma as cpu
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    cell, mf, nocc, _ = system
    mp, xy = _solve(system, mult)
    mf.get_k = _dense_get_k(cell)          # the CPU driver's 2-RDM exchange
    g_cpu = cpu.Gradients(mp, mf, mult, 0)
    g_gpu = gpu.Gradients(mp, mf, mult, 0)
    for g in (g_cpu, g_gpu):
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
    de_cpu = g_cpu.grad_elec(xy, mult, range(cell.natm))
    assert g_gpu.cphf_x0 is None
    de_gpu = g_gpu.grad_elec(xy, mult, range(cell.natm))
    scale = np.abs(de_cpu).max()
    assert np.abs(de_gpu - de_cpu).max() < 1e-7 * scale
    # the warm start seeds the second solve and leaves the result unchanged
    assert g_gpu.cphf_x0 is not None and g_gpu.cphf_x0.shape == (cell.nao, cell.nao)
    de_warm = g_gpu.grad_elec(xy, mult, range(cell.natm))
    assert np.abs(de_warm - de_gpu).max() < 1e-9 * scale


def test_cphf_chkfile_carries_the_warm_start_to_a_new_run(system, tmp_path, monkeypatch):
    from lib_pprpa.grad import pprpa as grad_pprpa
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    from lib_pprpa.grad.pprpa import load_cphf_x0
    cell, mf, nocc, _ = system
    mp, xy = _solve(system, "t")
    chk = str(tmp_path / "run.chk")
    loaded = []
    monkeypatch.setattr(grad_pprpa, "load_cphf_x0",
                        lambda *a: loaded.append(load_cphf_x0(*a)) or loaded[-1])
    de = []
    for _ in range(2):
        g = gpu.Gradients(mp, mf, "t", 0)
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
        g.cphf_chkfile = chk
        de.append(g.grad_elec(xy, "t", range(cell.natm)))
        assert np.array_equal(load_cphf_x0(chk, "t", 0), g.cphf_x0)
    # the first run found no record, the second started from the first's solution
    assert loaded[0] is None and loaded[1] is not None
    assert np.abs(de[1] - de[0]).max() < 1e-9 * np.abs(de[0]).max()
    assert load_cphf_x0(chk, "s", 0) is None


# omega = 1 / bohr: on this cell's mesh the long-range kernel keeps most of its weight
# (at HSE06's 0.11 it is ~exp(-35) at the smallest G and the test could not tell)
@pytest.mark.parametrize("xc", ("0.25*SR_HF(1.0) + 0.75*PBE, PBE",
                                "0.4*LR_HF(1.0) + 0.6*PBE, PBE", "pbe0"))
def test_hybrid_reference_exchange_matches_cpu(xc):
    """Hybrid reference exchange force against the CPU driver: short-range exchange only
    (alpha = 0), long-range only (hyb = 0), and PBE0 (the same factor at both ranges)."""
    from pyscf.pbc import dft as cdft
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    from lib_pprpa.grad import pprpa_gamma as cpu
    from lib_pprpa.grad import pprpa_gamma_gpu as gpu
    cell = _cell()
    mf = cdft.RKS(cell, xc=xc)
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    eri = gpu_ao2mo_blocks(cell, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:], cell.mesh)
    mp, xy = _solve((cell, mf, nocc, eri), "t")
    de = []
    for module in (cpu, gpu):
        g = module.Gradients(mp, mf, "t", 0)
        g.cphf_conv_tol = 1e-10
        g.cphf_max_cycle = 100
        de.append(g.grad_elec(xy, "t", range(cell.natm)))
    assert np.abs(de[1] - de[0]).max() < 1e-7 * np.abs(de[0]).max()
