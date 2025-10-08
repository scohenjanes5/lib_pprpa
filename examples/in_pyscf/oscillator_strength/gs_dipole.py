from pyscf import gto, dft
import numpy, pandas, os
from lib_pprpa.pyscf_util import get_pyscf_input_mol, read_dump_file
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from pyscf.lib import chkfile

molecules = {
    "water": [
        ("H", [-1.8235290184215478, 0.8842406482064837, 0.0]),
        ("H", [1.8235290184215478, 0.8842406482064837, 0.0]),
        ("O", [0.0, -0.11362923187009717, 0.0]),
    ],
    "methane": [
        ("C", [-16.25856106887544, 1.196102150543456, 0.0]),
        ("H", [-14.69112773485495, 0.7094598789454611, 1.304383457481034]),
        ("H", [-15.626504371992164, 2.69365341247753, -1.3239988146540191]),
        ("H", [-16.817542056521784, -0.5072024918332626, -1.0869326723273323]),
        ("H", [-17.899107906655352, 1.8884978025840944, 1.1065102349778262]),
    ],
    # "CO2": [
    #     ("C", [-14.629447074149146, 1.1848204855798024, -0.4782329903436802]),
    #     ("O", [-12.599389887573881, 0.9157423827029834, 0.47936682601841923]),
    #     ("O", [-16.659504260724407, 1.4539174857178672, -1.4358328067057795]),
    # ],
    # "CO": [
    #     ("C", [-14.951021768766383, 0.9779143722011738, -0.6524468417673333]),
    #     ("O", [-12.582439044236533, 1.035248662820478, 0.49833967630905246]),
    # ],
    "DME": [
        ("C", [2.933346274117363, -1.2044547400140335, -0.2596105749927482]),
        ("O", [1.047739752503853, -0.6706260070856491, 1.639620871978876]),
        ("H", [4.647856992412752, -1.8801263158522714, 0.6745944319472358]),
        ("H", [2.2276280528985404, -2.6617359382336265, -1.5434527094997599]),
        ("H", [3.3577598644334303, 0.5206951363626572, -1.3156651224446874]),
        ("C", [-1.3238476365640541, 0.24284870426785612, 0.6443210194317035]),
        ("H", [-2.6173273743063477, 0.5902559550078971, 2.2172156619521872]),
        ("H", [-1.0064114421596149, 2.003771096182563, -0.3895481433178418]),
        ("H", [-2.1369967879644, -1.178527697585001, -0.6163719700493863]),
    ],
    "CH4": [
        ("C", [-16.25856106887544, 1.196102150543456, 0.0]),
        ("H", [-14.69112773485495, 0.7094598789454611, 1.304383457481034]),
        ("H", [-15.626504371992164, 2.69365341247753, -1.3239988146540191]),
        ("H", [-16.817542056521784, -0.5072024918332626, -1.0869326723273323]),
        ("H", [-17.899107906655352, 1.8884978025840944, 1.1065102349778262]),
    ],
    # "acetic_acid": [
    #     ("C", [0.06619710614351411, 0.13927281538044506, 0.09645162139780077]),
    #     ("C", [-2.6319160599879896, -0.7945353490733802, 0.019747638001704895]),
    #     ("O", [0.6518232321462268, 2.3482681686907743, 0.27635354845639465]),
    #     ("O", [1.734220561774603, -1.6547764754978875, -0.17417605690116175]),
    #     ("H", [-3.1893852667346834, -1.1896392871974435, -1.9277474141913107]),
    #     ("H", [-2.828428679681511, -2.5314582192061112, 1.1187178657425165]),
    #     ("H", [-3.9078969338168106, 0.6251591965286137, 0.8056469386858228]),
    #     ("H", [3.3827231465389347, -0.835107768967792, -0.13966965786660374]),
    # ],
    "acetone": [
        ("C", [2.437935673301386, -1.282368148129851, -0.027608898679895553]),
        ("C", [0.0022865686107237247, 0.21539098367792575, -0.08653055924383418]),
        ("O", [-0.0033448152404801598, 2.5098775468635783, -0.08375266184072354]),
        ("C", [-2.4295452893083174, -1.2869979771350355, 0.023829446430765427]),
        ("H", [2.9126537730533753, -1.7840148451568922, 1.9171649478937467]),
        ("H", [2.2546511364798207, -3.020632723811023, -1.1254452907459682]),
        ("H", [3.9836749514118703, -0.17196507733542063, -0.8272276110283557]),
        ("H", [-2.0551527495094875, -3.314598519748364, 0.11544336894967963]),
        ("H", [-3.5148716944297695, -0.7546810251063031, 1.69738979960683]),
        ("H", [-3.5659698888380085, -0.9129266907773813, -1.6573276057660506]),
    ],
    "O2": [
        ("O", [-1.3511541790640191, 0.0, 0.0]),
        ("O", [1.3511541790640191, 0.0, 0.0]),
    ],
    "propane": [
        ("C", [2.4030324317806695, -0.5207896226688854, 0.0]),
        ("C", [0.0, 1.1197572151110273, 0.0]),
        ("C", [-2.4030324317806695, -0.5207896226688854, 0.0]),
        ("H", [2.4633902841992774, -1.7246207530618125, 1.6763193533179295]),
        ("H", [2.4633902841992774, -1.7246207530618125, -1.6763193533179295]),
        ("H", [4.084831990859837, 0.6757471648832205, 0.0]),
        ("H", [0.0, 2.3398777846977055, 1.6655857089303998]),
        ("H", [0.0, 2.3398777846977055, -1.6655857089303998]),
        ("H", [-2.4633902841992774, -1.7246207530618125, 1.6763193533179295]),
        ("H", [-4.084831990859837, 0.6757471648832205, 0.0]),
        ("H", [-2.4633902841992774, -1.7246207530618125, -1.6763193533179295]),
    ],
}
basis = "631g*"


def run_dft(key, charge=0):
    geometry = molecules[key]
    spins = [0, 2]
    energies = []
    mfs = []
    for spin in spins:
        mol = gto.Mole()
        mol.verbose = 0
        # The geoms were obtained in dict form from ._atom of loaded xyz files.
        # So coords are in Bohr
        mol.atom = geometry
        mol.unit = "Bohr"
        mol.basis = basis
        mol.charge = charge
        try:
            mol.spin = spin
            mol.build()
        except RuntimeError:
            energies.append(numpy.inf)
            mfs.append(None)
            print(
                f"Failed to build molecule {key} with charge {charge} and spin {spin}"
            )
            continue

        mf = dft.RKS(mol)
        mf.xc = "B3LYP"
        mf.chkfile = f"{key}_c{charge}_s{spin}.chk"
        if os.path.isfile(mf.chkfile):
            print(f"loading checkpoint data from {mf.chkfile}")
            data = chkfile.load(mf.chkfile, "scf")
            mf.__dict__.update(data)
        else:
            mf.kernel()
        energies.append(mf.e_tot)
        mfs.append(mf)

    index_min = numpy.argmin(energies)
    mf = mfs[index_min]
    print(f"Gs spin of {key} with charge {charge} is {mf.mol.spin}")
    if mf.mol.spin != 0 and charge != 0:
        print(
            f"Warning: The ground state of {key} is not a singlet, but a triplet state."
        )
        return None
    return mfs[index_min]


def run_calc(key, channel="pp"):
    if channel == "hh":
        charge = -2  # start from the N+2 electron system
    else:
        charge = 2  # start from the N-2 electron system
    mf = run_dft(key, charge=charge)
    if mf is None:
        return None, None

    h5_file = f"{key}.h5"

    if os.path.exists(h5_file):
        nocc, mo_energy, Lpq, mo_dip = read_dump_file(h5_file, with_dip=True)
    else:
        nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(
            mf, with_dip=True, dump_file=key
        )

    pprpa = ppRPA_Davidson(
        nocc, mo_energy, Lpq, mo_dip=mo_dip, channel=channel, nroot=1, trial="subspace"
    )
    pprpa._use_Lov = True
    pprpa.kernel("s")
    pprpa.kernel("t")
    pprpa.analyze()

    tdms = pprpa.tdm

    dipole = tdms[0]
    return dipole, mf.mol.spin


def get_dips(key="water"):
    dipolepp, spinpp = run_calc(key, channel="pp")
    dipolehh, spinhh = run_calc(key, channel="hh")
    mf_N = run_dft(key)
    spin_gs = mf_N.mol.spin

    dip = mf_N.dip_moment()
    ref = numpy.linalg.norm(dip) if dip is not None else numpy.nan
    pp = numpy.linalg.norm(dipolepp) if dipolepp is not None else numpy.nan
    hh = numpy.linalg.norm(dipolehh) if dipolehh is not None else numpy.nan

    return ref, spin_gs, pp, spinpp, hh, spinhh


refs = []
pps = []
hhs = []
names = []
spin_Ns = []
spin_Np2s = []
spin_Nm2s = []
for k in molecules.keys():
    print(f"*****Processing {k}*****")
    r, N, p, Nm2, h, Np2 = get_dips(k)
    refs.append(r)
    pps.append(p)
    hhs.append(h)
    spin_Ns.append(N)
    spin_Nm2s.append(Nm2)
    spin_Np2s.append(Np2)

df = pandas.DataFrame(
    {
        "Mol": molecules.keys(),
        "Ref": refs,
        "pp": pps,
        "hh": hhs,
        "spin_N": spin_Ns,
        "spin_Np2": spin_Np2s,
        "spin_Nm2": spin_Nm2s,
    }
)
print(df)
df.to_csv("dip_results.csv")
