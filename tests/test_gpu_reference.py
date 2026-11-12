"""The pp-RPA's Gamma KRKS reference: the ACE SCF ends on the full-exchange orbitals, and the
reference's DFT gradient matches pyscf's CPU KRKS gradient for a GGA and a global hybrid."""

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
HYBRID = "0.25*HF + 0.75*PBE, PBE"


def _cell(ke=100.0):
    """Neutral C2 diamond cell, one atom off its site."""
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.05, a0 / 4 - 0.03, a0 / 4]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.ke_cutoff = ke
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell


def test_semilocal_part_of_a_global_hybrid():
    from lib_pprpa.gpu_reference import semilocal_xc
    assert semilocal_xc("0.18*HF+0.82*PBE,PBE") == "0.82*PBE,PBE"
    assert semilocal_xc("0.82*PBE + 0.18*HF, PBE") == "0.82*PBE,PBE"
    with pytest.raises(ValueError):
        semilocal_xc("PBE,PBE")
    with pytest.raises(ValueError):
        semilocal_xc("HSE06")


def test_ace_reference_has_the_full_exchange_orbitals():
    """Energy, density and every orbital energy of the ACE SCF + full_exchange_orbitals equal
    those of gpu4pyscf's KRKS with the full FFTDF exchange."""
    import cupy as cp
    from gpu4pyscf.pbc import dft as gdft
    from lib_pprpa.gpu_fftdf import LowRankFFTDF
    from lib_pprpa.gpu_reference import make_krks, run_scf
    cell = _cell()
    ace = make_krks(cell, HYBRID, conv_tol=1e-11)
    assert isinstance(ace.with_df, LowRankFFTDF) and ace.with_df.ace
    run_scf(ace)
    ref = gdft.KRKS(cell, kpts=KPTS, xc=HYBRID)
    ref.exxdiv = None
    ref.conv_tol = 1e-11
    ref.kernel()
    assert abs(ace.e_tot - ref.e_tot) < 1e-9
    assert float(abs(ace.make_rdm1() - ref.make_rdm1()).max()) < 1e-6
    e_ace, e_ref = cp.asnumpy(ace.mo_energy[0]), cp.asnumpy(ref.mo_energy[0])
    assert np.abs(e_ace - e_ref).max() < 1e-7


@pytest.mark.parametrize("xc", ("pbe", HYBRID))
def test_dft_gradient_matches_cpu(xc):
    """The multigrid gradient integrates the xc on other grids than pyscf's FFTDF numint:
    the two agree to ~1e-8 Ha/Bohr at ke 200 (5e-5 at ke 100, 5e-10 at ke 400); the
    exact-exchange term alone matches to 3e-9 at ke 200."""
    from pyscf.pbc import dft as cdft
    from lib_pprpa.gpu_reference import dft_gradient, make_krks, run_scf
    cell = _cell(ke=200.0)
    kmf = make_krks(cell, xc, conv_tol=1e-11)
    run_scf(kmf)
    de = dft_gradient(kmf)
    ref = cdft.KRKS(cell, kpts=KPTS, xc=xc)
    ref.exxdiv = None
    ref.conv_tol = 1e-11
    ref.kernel()
    de_ref = np.asarray(ref.nuc_grad_method().kernel())
    assert abs(kmf.e_tot - ref.e_tot) < 1e-7
    assert de.shape == (cell.natm, 3)
    assert np.abs(de - de_ref).max() < 1e-6 * np.abs(de_ref).max()
