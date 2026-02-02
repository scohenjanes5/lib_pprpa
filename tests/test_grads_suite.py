import numpy as np
import pytest
from pyscf import gto, scf, dft, pbc
from lib_pprpa import pprpa_davidson, gpprpa_davidson
from lib_pprpa.grad import pprpa_gamma, grad_utils
from benchmarks import references as ref_implementations
from lib_pprpa.grad import pprpa as pprpa_grad_mod
from lib_pprpa.grad import pprpa_gamma as pprpa_gamma_grad_mod
from lib_pprpa.grad.ase_utils import pprpaobj
try:
    from lib_pprpa.grad import gpprpa as gpprpa_grad_mod
    HAS_SOCUTILS = True
except (ImportError, ModuleNotFoundError):
    gpprpa_grad_mod = None
    HAS_SOCUTILS = False

from test_utils import get_water_mol, get_water_cell, get_water_rks, get_water_rhf, get_water_ghf, get_water_pbc_rks

@pytest.fixture(scope="module")
def h2o_s_act():
    from lib_pprpa import pyscf_util
    mol, mf = get_water_rks(df=True)
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol(mf, nocc_act=5, nvir_act=5)
    pp = pprpaobj(mf, 'pp', Lpq=Lpq, AS_size=5, nroot=1)
    pp.kernel('s')
    return mol, mf, pp

def test_water_s_act_grad(h2o_s_act):
    mol, mf, pp = h2o_s_act
    g = pprpa_grad_mod.Gradients(pp, mf, mult='s')
    grad = g.kernel()
    ref = ref_implementations.grad_elec_ref(g, pp.xy_s[0], 's') + g.grad_nuc()
    assert np.allclose(grad, ref, atol=1e-8)

def test_water_s_act_intermediates(h2o_s_act):
    mol, mf, pp = h2o_s_act
    xy = pp.xy_s[0]
    dm0, i_int = pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa(pp, mf, xy=xy, mult='s')
    dm0_ref, i_int_ref = ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref(pp, mf, xy=xy, mult='s')
    
    assert np.allclose(dm0, dm0_ref, atol=1e-8)
    assert np.allclose(i_int, i_int_ref, atol=1e-8)
    
    mf_grad = mf.nuc_grad_method()
    dm0_hf = mf.make_rdm1()
    dm0_1_ao = mf.mo_coeff @ dm0 @ mf.mo_coeff.T
    
    vxc, vjk = grad_utils.get_veff_rks(mf_grad, mol, (dm0_hf, dm0_1_ao))
    f1vo, _, _, _ = grad_utils._contract_xc_kernel(mf, mf.xc, dm0_1_ao)
    
    vxc_ref, vjk_ref = ref_implementations.get_veff_rks_ref(mf_grad, mol, (dm0_hf, dm0_1_ao))
    f1vo_ref, _, _, _ = ref_implementations._contract_xc_kernel_ref(mf, mf.xc, dm0_1_ao)
    
    assert np.allclose(vxc, vxc_ref, atol=1e-8)
    assert np.allclose(vjk, vjk_ref, atol=1e-8)
    assert np.allclose(f1vo, f1vo_ref, atol=1e-8)

@pytest.fixture(scope="module")
def h2o_t_eri():
    mol, mf = get_water_rhf(df=False)
    pp = pprpaobj(mf, 'pp', mo_eri=True, nroot=1)
    pp.kernel('t')
    return mol, mf, pp

def test_water_t_noact_eri_grad(h2o_t_eri):
    mol, mf, pp = h2o_t_eri
    g = pprpa_grad_mod.Gradients(pp, mf, mult='t')
    grad = g.kernel()
    ref = ref_implementations.grad_elec_ref(g, pp.xy_t[0], 't') + g.grad_nuc()
    assert np.allclose(grad, ref, atol=1e-8)

def test_water_t_noact_eri_intermediates(h2o_t_eri):
    mol, mf, pp = h2o_t_eri
    xy = pp.xy_t[0]
    dm0, i_int = pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa(pp, mf, xy=xy, mult='t')
    dm0_ref, i_int_ref = ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref(pp, mf, xy=xy, mult='t')
    
    assert np.allclose(dm0, dm0_ref, atol=1e-8)
    assert np.allclose(i_int, i_int_ref, atol=1e-8)

@pytest.fixture(scope="module")
def h2o_t_act_large():
    from lib_pprpa import pyscf_util
    mol, mf = get_water_rhf(df=True)
    # Triplet water, custom nocc/nvir larger than system
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol(mf, nocc_act=10, nvir_act=10)
    pp = pprpaobj(mf, 'pp', Lpq=Lpq, AS_size=10, nroot=1)
    pp.kernel('t')
    return mol, mf, pp

def test_water_t_large_act_grad(h2o_t_act_large):
    mol, mf, pp = h2o_t_act_large
    g = pprpa_grad_mod.Gradients(pp, mf, mult='t')
    grad = g.kernel()
    ref = ref_implementations.grad_elec_ref(g, pp.xy_t[0], 't') + g.grad_nuc()
    assert np.allclose(grad, ref, atol=1e-8)

@pytest.fixture(scope="module")
def h2o_pbc():
    cell, mf = get_water_pbc_rks()
    # PBC must use MO-ERI or AO-direct with non-DF DFT
    pp = pprpaobj(mf, 'pp', mo_eri=True, nroot=1)
    pp.kernel('s')
    return cell, mf, pp

def test_h2o_pbc_grad(h2o_pbc):
    cell, mf, pp = h2o_pbc
    # pprpa_gamma.Gradients
    g = pprpa_gamma_grad_mod.Gradients(pp, mf, mult='s')
    grad = g.kernel()
    ref = ref_implementations.grad_elec_gamma_ref(g, pp.xy_s[0], 's') + g.grad_nuc()
    assert np.allclose(grad, ref, atol=1e-8)

def test_h2o_pbc_intermediates(h2o_pbc):
    cell, mf, pp = h2o_pbc
    xy = pp.xy_s[0]
    # grad_elec from pprpa_gamma
    g = pprpa_gamma_grad_mod.Gradients(pp, mf, mult='s')
    grad_elec = g.grad_elec(xy, 's', atmlst=range(cell.natm))
    grad_elec_ref = ref_implementations.grad_elec_gamma_ref(g, xy, 's', atmlst=range(cell.natm))
    assert np.allclose(grad_elec, grad_elec_ref, atol=1e-8)
    
    # Check RDM
    dm0, i_int = pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa(pp, mf, xy=xy, mult='s')
    dm0_ref, i_int_ref = ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref(pp, mf, xy=xy, mult='s')
    assert np.allclose(dm0, dm0_ref, atol=1e-8)
    assert np.allclose(i_int, i_int_ref, atol=1e-8)

@pytest.fixture(scope="module")
def water_triplet_ghf_hh():
    if not HAS_SOCUTILS:
        pytest.skip("socutils not installed")
    from lib_pprpa import pyscf_util
    mol, mf = get_water_ghf(df=True)
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol_g(mf)
    pp = pprpaobj(mf, 'hh', Lpq=Lpq, cls=gpprpa_davidson.GppRPA_Davidson, nroot=1)
    pp.kernel()
    return mol, mf, pp

@pytest.mark.skipif(not HAS_SOCUTILS, reason="socutils not installed")
def test_water_triplet_ghf_hh_intermediates(water_triplet_ghf_hh):
    mol, mf, pp = water_triplet_ghf_hh
    xy = pp.xy_t[0]
    dm0, i_int = gpprpa_grad_mod.make_rdm1_relaxed_ghf_pprpa(pp, mf, xy=xy)
    dm0_ref, i_int_ref = ref_implementations.make_rdm1_relaxed_ghf_pprpa_ref(pp, mf, xy=xy)
    
    assert np.allclose(dm0, dm0_ref, atol=1e-7)
    assert np.allclose(i_int, i_int_ref, atol=1e-7)

def test_water_s_cross_method_consistency():
    """Compare final gradients for use_eri=True, DF (Lpq), and AO-direct."""
    from lib_pprpa import pyscf_util
    # Setup 1: DF-DFT + Density Fitting (Lpq)
    mol_df, mf_df = get_water_rks(df=True)
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol(mf_df)
    pp_df = pprpaobj(mf_df, 'pp', Lpq=Lpq, nroot=1)
    pp_df.kernel('s')
    grad_df = pprpa_grad_mod.Gradients(pp_df, mf_df, mult='s').kernel()
    
    # Setup 2: no-DF-DFT + MO-ERI
    mol_eri, mf_eri = get_water_rks(df=False)
    pp_eri = pprpaobj(mf_eri, 'pp', mo_eri=True, nroot=1)
    pp_eri.kernel('s')
    grad_eri = pprpa_grad_mod.Gradients(pp_eri, mf_eri, mult='s').kernel()
    
    # Setup 3: no-DF-DFT + AO-direct
    mol_ao, mf_ao = get_water_rks(df=False)
    pp_ao = pprpaobj(mf_ao, 'pp', mo_eri=False, Lpq=None, nroot=1)
    pp_ao.kernel('s')
    grad_ao = pprpa_grad_mod.Gradients(pp_ao, mf_ao, mult='s').kernel()
    
    # Comparisons
    # DF vs others might have slight numerical differences due to DFT grid/fit
    assert np.allclose(grad_df, grad_eri, atol=1e-3)
    assert np.allclose(grad_df, grad_ao, atol=1e-3)
    assert np.allclose(grad_eri, grad_ao, atol=1e-8)

def test_h2o_pbc_cross_method_consistency():
    """Compare final gradients for PBC: (no-DF-DFT + MO-ERI) vs (no-DF-DFT + AO-direct)."""
    cell, mf = get_water_pbc_rks()
    
    # Setup 1: MO-ERI
    pp_eri = pprpaobj(mf, 'pp', mo_eri=True, nroot=1)
    pp_eri.kernel('s')
    grad_eri = pprpa_gamma_grad_mod.Gradients(pp_eri, mf, mult='s').kernel()
    
    # Setup 2: AO-direct
    pp_ao = pprpaobj(mf, 'pp', mo_eri=False, Lpq=None, nroot=1)
    pp_ao.kernel('s')
    grad_ao = pprpa_gamma_grad_mod.Gradients(pp_ao, mf, mult='s').kernel()
    
    # Comparisons
    assert np.allclose(grad_eri, grad_ao, atol=1e-3)
