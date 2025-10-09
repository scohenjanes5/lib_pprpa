from pyscf import gto, dft, scf
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
    "isopropanol": [
        ("C", [0.017839014615894183, 0.01952087086675709, 0.6960239261998036]),
        ("C", [-1.22032843946038, -2.455018797467454, -0.21032651766409138]),
        ("C", [2.786495656977412, 0.1947740716589209, -0.1875553178630824]),
        ("O", [-1.3585808027335597, 2.1193089514384713, -0.2963090563318017]),
        ("H", [-0.037208707392686065, 0.08944073747566438, 2.7610599433407663]),
        ("H", [-3.175628060547849, -2.5658701319344406, 0.44281952276933095]),
        ("H", [-1.2153017679690368, -2.5672685292666193, -2.272754712753154]),
        ("H", [-0.20084009251877477, -4.088535854063985, 0.5339421164958583]),
        ("H", [2.9045279507177457, 0.15240641194617224, -2.2487929854936692]),
        ("H", [3.644015577782546, 1.9515957578833218, 0.476513339570326]),
        ("H", [3.887620172500228, -1.382107892984395, 0.5625903645442645]),
        ("H", [-3.0924234192832496, 2.0278462070095222, 0.3181920848542651]),
    ],
    "formaldehyde": [
        ("C", [1.1508621071213685, 0.0, 0.001341705548441194]),
        ("O", [-1.153583312740742, 0.0, -0.0005291233148782173]),
        ("H", [2.175471611860545, 1.7746984953627865, 0.0021542877820041706]),
        ("H", [2.175471611860545, -1.7746984953627865, 0.0021542877820041706]),
    ],
    "isobutane": [
        ("C", [0.047526612032811305, -0.026456165743910867, 0.6816242131306178]),
        ("C", [-1.170458567033108, -2.522727684510621, -0.20649037363122433]),
        ("C", [2.8290333920413717, 0.12190623229569214, -0.1824530573267567]),
        ("C", [-1.45777252701198, 2.2594699380974617, -0.3246171536777863]),
        ("H", [0.003420404285462762, 0.02173185043249821, 2.746093312434211]),
        ("H", [-3.1217708659977452, -2.6525329720069943, 0.45563186589388205]),
        ("H", [-1.174086841192273, -2.6487157252353732, -2.2679359111355133]),
        ("H", [-0.13267767120571297, -4.143111144541424, 0.5419356580027684]),
        ("H", [2.964262193515247, 0.08014328494280427, -2.242689170111324]),
        ("H", [3.704902553516032, 1.865877780873051, 0.49123430608068785]),
        ("H", [3.908615029744146, -1.46793925356214, 0.5720389951670898]),
        ("H", [-1.464481054754186, 2.2858883093188815, -2.389483095467538]),
        ("H", [-3.414281572819171, 2.1844100164297373, 0.3307020717988858]),
        ("H", [-0.6267276692120027, 4.026760707051953, 0.34527186021928247]),
    ],
    # "ethanol": [
    #     ("C", [0.0488305230587612, -1.057793095486539, 0.0]),
    #     ("C", [-2.4220241793325483, 0.47029614062050695, 0.0]),
    #     ("O", [2.1508673777187077, 0.6340220120528238, 0.0]),
    #     ("H", [0.12498648587873319, -2.2557093831095774, 1.6821397097815898]),
    #     ("H", [0.12498648587873319, -2.2557093831095774, -1.6821397097815898]),
    #     ("H", [-2.5325164658358674, 1.6704234078092863, 1.676281558795438]),
    #     ("H", [-2.5325164658358674, 1.6704234078092863, -1.676281558795438]),
    #     ("H", [-4.048360276655732, -0.8003179110145493, 0.0]),
    #     ("H", [3.70208686159043, -0.3590290664061161, 0.0]),
    # ],
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
        return None

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
    return dipole


def run_HF(key, spin=0):
    geometry = molecules[key]
    mol = gto.Mole()
    mol.verbose = 0
    # The geoms were obtained in dict form from ._atom of loaded xyz files.
    # So coords are in Bohr
    mol.atom = geometry
    mol.unit = "Bohr"
    mol.basis = basis
    mol.charge = 0
    mol.spin = spin
    mol.build()

    mf = scf.RHF(mol)
    mf.xc = "HF"
    mf.chkfile = f"{key}_HF_s{spin}.chk"
    if os.path.isfile(mf.chkfile):
        print(f"loading checkpoint data from {mf.chkfile}")
        data = chkfile.load(mf.chkfile, "scf")
        mf.__dict__.update(data)
    else:
        mf.kernel()
    print(f"HF spin of {key} is {mf.mol.spin}")
    return mf.dip_moment(unit="AU")


def get_dips(key="water"):
    dipolepp = run_calc(key, channel="pp")
    dipolehh = run_calc(key, channel="hh")
    mf_N = run_dft(key)
    spin_gs = mf_N.mol.spin
    hf_dip = run_HF(key, spin=spin_gs)

    dip = mf_N.dip_moment(unit="AU")
    ref = numpy.linalg.norm(dip) if dip is not None else numpy.nan
    pp = numpy.linalg.norm(dipolepp) if dipolepp is not None else numpy.nan
    hh = numpy.linalg.norm(dipolehh) if dipolehh is not None else numpy.nan
    hf = numpy.linalg.norm(hf_dip) if hf_dip is not None else numpy.nan

    return ref, pp, hh, hf


refs = []
hf_results = []
xxRPA = []
channels = []
for k in molecules.keys():
    print(f"*****Processing {k}*****")
    r, p, h, hf = get_dips(k)
    refs.append(r)
    xx_modes = {"pp": p, "hh": h}
    xx_data = [v.round(4) for v in xx_modes.values() if v is not numpy.nan]
    modes = [k for k, v in xx_modes.items() if v is not numpy.nan]
    xxRPA.append(xx_data if len(xx_data) > 1 else xx_data[0])
    channels.append(modes if len(modes) > 1 else modes[0])
    hf_results.append(hf)

df = pandas.DataFrame(
    {
        "Mol": molecules.keys(),
        "B3LYP": refs,
        "HF": hf_results,
        "xxRPA": xxRPA,
        "Channel": channels,
    }
)
print(df.round(4))
df.to_csv("dip_results.csv")
