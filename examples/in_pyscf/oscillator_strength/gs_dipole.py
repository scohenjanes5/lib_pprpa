from pyscf import gto, scf, dft
import numpy, pandas
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.pprpa_davidson import ppRPA_Davidson


molecules = {"water": [
        ["O", (0.00000000, -0.00000000, -0.00614048)],
        ["H", (0.76443318, -0.00000000, 0.58917024)],
        ["H", (-0.76443318, 0.00000000, 0.58917024)],
    ],
    "methane": "methane.xyz",
    "CO2": "CO2.xyz",
    "CO": "CO.xyz",
    "DME": "dimethylether.xyz",
    "CH4": "methane.xyz"}
basis = "ccpvdz"

def run_calc(geometry, channel="pp"):
    mol = gto.Mole()
    mol.verbose = 0

    mol.atom = geometry

    mol.basis = basis
    if channel == "hh":
        mol.charge = -2  # start from the N+2 electron system
    else:
        mol.charge = 2  # start from the N-2 electron system
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = "B3LYP"
    mf.kernel()

    nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)

    pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq, mo_dip=mo_dip, channel=channel)
    pprpa.kernel("s")
    pprpa.kernel("t")
    pprpa.analyze()

    tdms = pprpa.tdm

    dipole = tdms[0]
    return dipole

def get_dips(key="water"):
    mol = gto.Mole()
    mol.verbose = 0
    geometry = molecules[key]
    mol.atom = geometry
    mol.basis = basis
    mol.build()
    mf = dft.RKS(mol)
    mf.kernel()
    dip = mf.dip_moment()
    ref = numpy.linalg.norm(dip)

    dipolepp = run_calc(geometry, channel="pp")
    dipolehh = run_calc(geometry, channel="hh")

    pp = numpy.linalg.norm(dipolepp)
    hh = numpy.linalg.norm(dipolehh)

    return ref, pp, hh

refs = []
pps = []
hhs = []
names = []
for k in molecules.keys():
    r, p, h = get_dips(k)
    refs.append(r)
    pps.append(p)
    hhs.append(h)

df = pandas.DataFrame({"Mol":molecules.keys(), "Ref":refs, "pp":pps, "hh":hhs})
print(df)
df.to_csv("dip_results.csv")

quit()
print("*******Ground State RKS B3LYP results*******")
print("Reference state dipole moment (a.u.):", dip)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dip) )
#print("Reference state dipole moment (Debye):", dip * 2.541746)
#print("Dipole Magnitude (Debye):", numpy.linalg.norm(dip) * 2.541746)
print()
print("*******pp-channel results*******")
print("Dipole moment (a.u.):", dipolepp)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dipolepp) )
#print("Dipole moment (Debye):", dipolepp * 2.541746)
#print("Dipole Magnitude (Debye):", numpy.linalg.norm(dipolepp) * 2.541746)
print()
print("*******hh-channel results*******")
print("Dipole moment (a.u.):", dipolehh)
print("Dipole Magnitude (a.u.):", numpy.linalg.norm(dipolehh) )
#print("Dipole moment (Debye):", dipolehh * 2.541746)
#print("Dipole Magnitude (Debye):", numpy.linalg.norm(dipolehh) * 2.541746)
