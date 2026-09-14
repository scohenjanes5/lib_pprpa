"""End-to-end GPU pp-RPA gradient on the C2 diamond cell: low-rank FFT pairing-K
(default) vs the AFT kernel, one slot vs a virtual two-slot group (which also
exercises the per-atom hcore dispatch and the low-rank exchange rows), and vs
the CPU reference gradient.  Triplet and singlet.

Mirrors examples/gpu/pbc_pprpa_gamma_opt_gpu.py (AO-direct Davidson).
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _need_cupy():
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDevice()
    except Exception:
        if pytest is None:
            raise RuntimeError("CUDA GPU required for this test") from None
        pytest.skip("CUDA GPU required")


def _two_slots():
    """Two real devices when the allocation has them, else two virtual slots on GPU 0."""
    import cupy as _cp
    return [0, 1] if _cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]


XC = "pbe"
CHANNEL = "pp"
CHARGE = +2
KPTS = np.zeros((1, 3))


def _build_cell():
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4, a0 / 4]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.charge = CHARGE
    cell.ke_cutoff = 100.0
    cell.verbose = 0
    cell.build()
    return cell


def _scf_and_pprpa(cell, mult):
    import cupy as cp
    from pyscf.pbc import dft as cdft
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.grad.ase_utils import pprpaobj
    from lib_pprpa.pprpa_davidson_gpu import attach_gpu_contraction
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
    mp = pprpaobj(mf, CHANNEL, nroot=2, mo_eri=False, nfrozen_occ=0, vir_cut=1e5)
    mp.residue_thresh = 1e-10
    attach_gpu_contraction(mp, kg)
    mp.kernel(mult)
    xy = (mp.xy_s if mult == "s" else mp.xy_t)[0]
    return kg, mf, mp, np.array(xy, copy=True)


def _patch_getk(mf, cell, group):
    import cupy as cp
    from gpu4pyscf.pbc.df.fft import FFTDF
    from gpu4pyscf.pbc.df import fft_jk
    from lib_pprpa.gpu_fft_k import attach_lowrank_getk
    fdf = FFTDF(cell, KPTS)

    def gk(dm=None, hermi=1, **kw):
        dmg = cp.asarray(dm)
        single = dmg.ndim == 2
        if single:
            dmg = dmg[None]
        K = cp.asnumpy(fft_jk.get_k(fdf, dmg, hermi=0, kpt=np.zeros(3), exxdiv=None))
        return K[0] if single else K
    mf.get_k = gk
    attach_lowrank_getk(mf, cell, cell.mesh, group=group)


def _gpu_grad(mp, mf, cell, mult, xy, method, group):
    from lib_pprpa.grad import pprpa_gamma  # noqa: F401
    from lib_pprpa.grad import pprpa_gamma_gpu as gpugrad
    _patch_getk(mf, cell, group)
    g = gpugrad.Gradients(mp, mf, mult, 0)
    g.cphf_conv_tol = 1e-10
    g.cphf_max_cycle = 100
    g.pairing_k_method = method
    g.gpu_group = group
    return g.grad_elec(xy, mult, range(cell.natm))


def _cpu_grad(mp, mf, cell, mult, xy):
    from lib_pprpa.grad import pprpa_gamma as cpu
    from lib_pprpa.gpu_multi import DeviceGroup
    _patch_getk(mf, cell, DeviceGroup([0]))
    g = cpu.Gradients(mp, mf, mult, 0)
    g.cphf_conv_tol = 1e-10
    g.cphf_max_cycle = 100
    return g.grad_elec(xy, mult, range(cell.natm))


def _run(mult):
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.grad import pprpa_gamma_gpu as gpugrad
    cell = _build_cell()
    kg, mf, mp, xy = _scf_and_pprpa(cell, mult)
    de_aft = _gpu_grad(mp, mf, cell, mult, xy, "aft", DeviceGroup([0]))
    de_lr1 = _gpu_grad(mp, mf, cell, mult, xy, "lowrank", DeviceGroup([0]))
    tel1 = gpugrad.get_last_telemetry()
    de_lr2 = _gpu_grad(mp, mf, cell, mult, xy, "lowrank", DeviceGroup(_two_slots()))
    tel2 = gpugrad.get_last_telemetry()
    de_cpu = _cpu_grad(mp, mf, cell, mult, xy)
    scale = max(np.abs(de_cpu).max(), 1e-3)
    d_aft = np.abs(de_lr1 - de_aft).max() / scale
    d_slots = np.abs(de_lr1 - de_lr2).max() / scale
    d_cpu = np.abs(de_lr1 - de_cpu).max() / scale
    print(f"mult={mult}: |de|max={scale:.3e}  lowrank-vs-aft={d_aft:.2e}  "
          f"1slot-vs-2slots={d_slots:.2e}  lowrank-vs-cpu={d_cpu:.2e}  "
          f"pairing_k s: aft={tel1['timings']['pairing_k_seconds']:.2f} "
          f"lowrank={tel2['timings']['pairing_k_seconds']:.2f}", flush=True)
    assert tel1["pairing_k_method"] == "lowrank" and tel2["gpu_slots"] == _two_slots()
    assert d_aft < 1e-6, d_aft          # AFT vs FFT agree to ~1e-8 (documented)
    assert d_slots < 1e-9, d_slots
    assert d_cpu < 1e-7, d_cpu
    return de_lr1


def test_triplet():
    _need_cupy()
    _run("t")


def test_singlet():
    _need_cupy()
    _run("s")


if __name__ == "__main__":
    _need_cupy()
    _run("t")
    print("OK  triplet")
    _run("s")
    print("OK  singlet")
