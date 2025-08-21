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
kmf.chkfile = chkfname

if os.path.isfile(chkfname):
    print("loading checkpoint data")
    data = chkfile.load(chkfname, "scf")
    kmf.__dict__.update(data)
else:
    kmf.kernel() # run SCF and create checkpoint file
    # An OOM error is expected from current PySCF
