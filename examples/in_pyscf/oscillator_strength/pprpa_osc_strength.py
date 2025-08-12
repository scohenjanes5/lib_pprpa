from pyscf import gto, scf

from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

mol = gto.Mole()
mol.verbose = 0

mol.atom = [
    ["O", (0.00000000, -0.00000000, -0.00614048)],
    ["H", (0.76443318, -0.00000000, 0.58917024)],
    ["H", (-0.76443318, 0.00000000, 0.58917024)],
]

mol.basis = "def2svp"
mol.charge = 2  # start from the N-2 electron system
mol.build()
mf = scf.RHF(mol)
mf.kernel()

nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)

pp_RPA_functions = [
    ppRPA_Davidson,
    ppRPA_direct,
]

for ppRPA in pp_RPA_functions:
    print(f"Testing {ppRPA.__name__}...")
    try:
        pprpa = ppRPA(nocc, mf.mo_energy, Lpq, mo_dip=mo_dip, hh_state=0)
    except TypeError:
        pprpa = ppRPA(nocc, mf.mo_energy, Lpq, mo_dip=mo_dip)
    pprpa.kernel("s")
    pprpa.kernel("t")
    pprpa.analyze()
