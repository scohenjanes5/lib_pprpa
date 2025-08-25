import os, h5py
import numpy as np
from pyscf.pbc import gto as pbcgto, df as pbc_df, dft as pbc_dft
from pyscf.pbc.lib import chkfile
from lib_pprpa.pyscf_util import get_pyscf_input_sc
from lib_pprpa.pprpa_util import generate_spectrum
from lib_pprpa.pprpa_davidson import ppRPA_Davidson


# GS of NV center, with centered defect
cell = pbcgto.Cell()
cell.unit = "A"
cell.a = np.array([[7.136, 0.   , 0.   ],
       [0.   , 7.136, 0.   ],
       [0.   , 0.   , 7.136]])
cell.atom = [
    ('C', np.array([0.90607351, 2.66432356, 2.69433958])),
    ('C', np.array([0.89409063, 2.68486665, 6.24190932])),
    ('C', np.array([0.89908438, 6.23582974, 2.67929897])),
    ('C', np.array([0.89243499, 6.24429479, 6.24356495])),
    ('C', np.array([4.45904861, 2.71945743, 2.67695115])),
    ('C', np.array([4.44166018, 2.66432356, 6.22992643])),
    ('C', np.array([4.46004038, 6.25473483, 2.67595939])),
    ('C', np.array([4.4567008 , 6.23582974, 6.23691556])),
    ('C', np.array([ 0.00012137,  0.00012137, -0.00012137])),
    ('C', np.array([-0.01010267, -0.01010267,  3.57551732])),
    ('C', np.array([-0.01010267,  3.56048268,  0.01010267])),
    ('C', np.array([0.00842563, 3.56246877, 3.57353123])),
    ('C', np.array([ 3.56048268, -0.01010267,  0.01010267])),
    ('C', np.array([3.56246877, 0.00842563, 3.57353123])),
    ('C', np.array([ 3.56246877,  3.56246877, -0.00842563])),
    ('C', np.array([0.89243499, 0.89243499, 0.89170515])),
    ('C', np.array([0.89409063, 0.89409063, 4.45113311])),
    ('C', np.array([0.89908438, 4.4567008 , 0.90017021])),
    ('C', np.array([0.90607351, 4.44166018, 4.4716762 ])),
    ('C', np.array([4.4567008 , 0.89908438, 0.90017021])),
    ('C', np.array([4.44166018, 0.90607351, 4.4716762 ])),
    ('C', np.array([4.46004038, 4.46004038, 0.88126512])),
    ('C', np.array([4.45904861, 4.45904861, 4.41654234])),
    ('C', np.array([-0.00690166,  1.78690562,  1.78056976])),
    ('C', np.array([2.51295667e-03, 1.78492731e+00, 5.35107257e+00])),
    ('C', np.array([0.00671939, 5.34920604, 1.78679384])),
    ('C', np.array([-0.00690166,  5.35543012,  5.34909426])),
    ('C', np.array([3.55769982, 1.80109051, 1.79992729])),
    ('C', np.array([3.62344369, 1.70724028, 5.4287596 ])),
    ('C', np.array([3.57026872, 5.36376987, 1.77223001])),
    ('C', np.array([3.55769982, 5.33607259, 5.33490937])),
    ('C', np.array([2.67502736, 2.67502736, 0.8924286 ])),
    ('C', np.array([2.67577993, 6.24343665, 0.89256329])),
    ('C', np.array([2.67502736, 6.24357134, 4.4609724 ])),
    ('C', np.array([6.24343665, 2.67577993, 0.89256329])),
    ('C', np.array([6.24357134, 2.67502736, 4.4609724 ])),
    ('C', np.array([6.24430677, 6.24430677, 0.89169317])),
    ('C', np.array([6.24343665, 6.24343665, 4.46021984])),
    ('C', np.array([ 1.78690562, -0.00690166,  1.78056976])),
    ('C', np.array([1.78492731e+00, 2.51295667e-03, 5.35107257e+00])),
    ('C', np.array([1.80109051, 3.55769982, 1.79992729])),
    ('C', np.array([1.70724028, 3.62344369, 5.4287596 ])),
    ('C', np.array([5.34920604, 0.00671939, 1.78679384])),
    ('C', np.array([ 5.35543012, -0.00690166,  5.34909426])),
    ('C', np.array([5.36376987, 3.57026872, 1.77223001])),
    ('C', np.array([5.33607259, 3.55769982, 5.33490937])),
    ('C', np.array([2.66432356, 0.90607351, 2.69433958])),
    ('C', np.array([2.68486665, 0.89409063, 6.24190932])),
    ('C', np.array([2.71945743, 4.45904861, 2.67695115])),
    ('C', np.array([2.66432356, 4.44166018, 6.22992643])),
    ('C', np.array([6.23582974, 0.89908438, 2.67929897])),
    ('C', np.array([6.24429479, 0.89243499, 6.24356495])),
    ('C', np.array([6.25473483, 4.46004038, 2.67595939])),
    ('C', np.array([6.23582974, 4.4567008 , 6.23691556])),
    ('C', np.array([ 1.78492731,  1.78492731, -0.00251296])),
    ('C', np.array([1.70724028, 1.70724028, 3.51255631])),
    ('C', np.array([1.78690562, 5.35543012, 0.00690166])),
    ('C', np.array([1.80109051, 5.33607259, 3.57830018])),
    ('C', np.array([5.35543012, 1.78690562, 0.00690166])),
    ('C', np.array([5.33607259, 1.80109051, 3.57830018])),
    ('C', np.array([ 5.34920604,  5.34920604, -0.00671939])),
    ('C', np.array([5.36376987, 5.36376987, 3.56573128])),
    ('N', np.array([3.66189577, 3.66189577, 3.47410423]))
]
cell.basis = 'ccpvdz'
cell.spin = 0
# for hh use cell.charge = -3
cell.charge = 1
cell.verbose = 5
cell.max_memory = 300000
cell.precision = 1e-10
cell.build()
gdf = pbc_df.RSDF(cell)
gdf.auxbasis = "cc-pvdz-ri"

scratch = "." # change to suitable scratch space on your system
gdf_filename = "3A2-geo.h5"
gdf_filename = os.path.join(scratch, gdf_filename) if os.path.exists(scratch) else gdf_filename
if not os.path.exists(gdf_filename):
    gdf._cderi_to_save = gdf_filename
    gdf.build()
else:
    print("using previous gdf data")

chkfname = "3A2-geo.chk"
kmf = pbc_dft.RKS(cell).rs_density_fit()
kmf.exxdiv = None
kmf.xc = "PBE0"
kmf.grids.level = 5
kmf.with_df = gdf
kmf.with_df._cderi = gdf_filename
kmf.with_df.omega = 0.5

if os.path.isfile(chkfname):
    print("loading checkpoint data")
    data = chkfile.load(chkfname, "scf")
    kmf.__dict__.update(data)
else:
    raise FileNotFoundError("Unable to find the checkpoint file")

dump_file = "NV-dump.h5"
dump_file = os.path.join(scratch, dump_file) if os.path.exists(scratch) else dump_file
if os.path.exists(dump_file):
    with h5py.File(dump_file, "r") as feri:
        nocc_act = int(np.asarray(feri["nocc"]))
        mo_energy_act = np.asarray(feri["mo_energy"])
        Lpq = np.asarray(feri["Lpq"])
        mo_dip = np.asarray(feri["mo_dipole"])
else:
    nocc_act, mo_energy_act, Lpq, mo_dip = get_pyscf_input_sc(kmf, nocc_act=300, nvir_act=300, dump_file=dump_file, with_dip=True)

# for hh-RPA use channel = "hh", and rename the spectrum file to something more appropriate
pprpa = ppRPA_Davidson(nocc_act, mo_energy_act, Lpq, nroot=20, trial="subspace", channel="pp", max_vec=1000, mo_dip=mo_dip, spectrum="pp")
pprpa._use_Lov = True
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# Plot spectrum
# This filename comes from the "spectrum" kwarg of the pprpa object
data = np.load("pp.npz")
energies = np.abs(data["vee"])
tdms = data["tdm"]
spectrum = generate_spectrum(energies, tdm=tdms, save_to="pp_spectrum.npz")

# You can plot if running interactively,
# or just readd the data from the npz file later to plot
# import matplotlib.pyplot as plt
# plt.plot(*spectrum)
# plt.show()