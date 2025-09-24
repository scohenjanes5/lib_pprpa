from pyscf import gto, scf
import numpy
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

geometry = [
        ["O", (0.00000000, -0.00000000, -0.00614048)],
        ["H", (0.76443318, -0.00000000, 0.58917024)],
        ["H", (-0.76443318, 0.00000000, 0.58917024)],
    ]
basis = "def2svp"
def run_calc(channel="pp"):
    mol = gto.Mole()
    mol.verbose = 0

    mol.atom = geometry

    mol.basis = basis
    if channel == "hh":
        mol.charge = -2  # start from the N+2 electron system
    else:
        mol.charge = 2  # start from the N-2 electron system
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)

    pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq, mo_dip=mo_dip, channel=channel)
    pprpa.kernel("s")
    pprpa.kernel("t")
    pprpa.analyze()

    tdms = pprpa.tdm

    dipole = tdms[0]
    return dipole

mol = gto.Mole()
mol.verbose = 0
mol.atom = geometry
mol.basis = basis
mol.build()
mf = scf.RHF(mol)
mf.kernel()
# calculate dipole moment of the reference state
dip = mf.dip_moment()

dipolepp = run_calc(channel="pp")
dipolehh = run_calc(channel="hh")


print("Reference state dipole moment (a.u.):", dip)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dip) )
print("Reference state dipole moment (Debye):", dip * 2.541746)
print("Dipole Magnitude (Debye):", numpy.linalg.norm(dip) * 2.541746)
print()
print("Dipole moment (a.u.):", dipolepp)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dipolepp) )
print("Dipole moment (Debye):", dipolepp * 2.541746)
print("Dipole Magnitude (Debye):", numpy.linalg.norm(dipolepp) * 2.541746)
print()
print("Dipole moment hh (a.u.):", dipolehh)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dipolehh) )
print("Dipole moment (Debye):", dipolehh * 2.541746)
print("Dipole Magnitude (Debye):", numpy.linalg.norm(dipolehh) * 2.541746)