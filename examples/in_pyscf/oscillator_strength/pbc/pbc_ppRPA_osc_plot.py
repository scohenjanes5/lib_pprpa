from libdmet.utils.misc import read_poscar
from pyscf.pbc import gto as pbcgto, scf as pbc_scf, df as pbc_df, dft as pbc_dft
from pyscf.pbc.lib import chkfile
import os, h5py
from lib_pprpa.pyscf_util import get_pyscf_input_sc, get_pyscf_input_mol
import numpy as np
from lib_pprpa.pprpa_davidson import ppRPA_Davidson


# GS of NV center, with centered defect
poscar_name = '3A2-geo.POSCAR.vasp'
cell = read_poscar(poscar_name)
cell.basis = 'ccpvdz'
cell.spin = 0
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

chkfname = "3A2-geo-pp.chk"
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

dump_file = "NV-dump-pp.h5"
dump_file = os.path.join(scratch, dump_file) if os.path.exists(scratch) else dump_file
if os.path.exists(dump_file):
    with h5py.File(dump_file, "r") as feri:
        nocc_act = int(np.asarray(feri["nocc"]))
        mo_energy_act = np.asarray(feri["mo_energy"])
        Lpq = np.asarray(feri["Lpq"])
        mo_dip = np.asarray(feri["mo_dipole"])
else:
    nocc_act, mo_energy_act, Lpq, mo_dip = get_pyscf_input_sc(kmf, nocc_act=300, nvir_act=300, dump_file=dump_file, with_dip=True)

pprpa = ppRPA_Davidson(nocc_act, mo_energy_act, Lpq, nroot=20, trial="subspace", channel="pp", max_vec=1000, mo_dip=mo_dip, spectrum="pp")
pprpa._use_Lov = True
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

