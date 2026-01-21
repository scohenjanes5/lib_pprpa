import numpy as np
from pyscf.pbc import gto
from lib_pprpa.grad import ase_utils
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

cell = gto.M(
    atom='''
O 0.000000000000   0.000000000000   0.000000000000
O 1.1 0.0 0.0
''',
    basis='gth-szv',
    pseudo='gth-pbe', # pseudopotential is required for PBC pprpa optimization
    a=np.eye(3)*5.0,
    charge=2,
    verbose=4,
)

def pprpaobj(mf, channel, nocc=None, nvir=None, mo_eri=False):
    mo_ene = mf.mo_energy
    mol = mf.mol
    if nocc is None:
        nocc = mol.nelectron // 2
    if nvir is None:
        nvir = mo_ene.shape[0] - nocc
    nmo = nocc + nvir
    mo_energy = mo_ene[mol.nelectron//2 - nocc:mol.nelectron//2 + nvir]
    mo_coeff = mf.mo_coeff[:,mol.nelectron//2 - nocc:mol.nelectron//2 + nvir]
    
    pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq=None, channel=channel, nroot=1)
    pprpa.cell = mol

    # One can use either the MO eri or the ao direct approach.
    # For small active spaces, MO eri should be faster.
    if mo_eri:
        eri = mf.with_df.get_mo_eri(mo_coeff, compact=False)
        eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
        vvvv = np.ascontiguousarray(eri[nocc:, nocc:, nocc:, nocc:])
        oovv = np.ascontiguousarray(eri[:nocc, :nocc, nocc:, nocc:])
        oooo = np.ascontiguousarray(eri[:nocc, :nocc, :nocc, :nocc])
        pprpa.use_eri(vvvv, oovv, oooo)
    else:
        pprpa._ao_direct = True
        pprpa._scf = mf
    
    pprpa.mu = 0.0
    return pprpa


def pprpa_grad(cell):
    mf = cell.RKS(xc="pbe")
    mf.chkfile = "mf.chk"
    mf.exxdiv = None
    mf.init_guess = "chkfile" # read dm from last step
    mf.kernel()
    e = mf.e_tot

    istate = 0
    mult = 't'
    nroot = 3
    mp = pprpaobj(mf, "pp")
    mp.nroot = nroot
    mp.kernel(mult)
    mp.analyze()
    e_pprpa = mp.exci_s[istate] if mult == 's' else mp.exci_t[istate]
    e = e + e_pprpa if mp.channel == "pp" else e - e_pprpa
    from lib_pprpa.grad import pprpa_gamma
    mpg = mp.Gradients(mf, mult, istate)
    mpg.kernel()
    return e, mpg.de


ase_utils.kernel(cell, pprpa_grad)
