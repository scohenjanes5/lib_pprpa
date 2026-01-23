import numpy as np
from pyscf.pbc import dft as pbc_dft
import pyscf
from lib_pprpa.pyscf_util import get_pyscf_input_sc
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.analyze import get_dipole_moment


def mfobj():
    cell = pyscf.M(
        atom=[
            ["C", [0.0, 0.0, 0.0]],
            ["O", [1.685068664391, 1.685068664391, 1.695068664391]],
        ],
        a=np.eye(3) * 5.0,
        unit="bohr",
        basis="gth-szv",
        pseudo="gth-pade",
        precision=1e-12,
        verbose=4,
    )
    mf = pbc_dft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-12
    return mf


mf = mfobj()
mf.kernel()
# consider adding nocc_act=AS_SIZE, nvir_act=AS_SIZE if your system is large
nocc_act, mo_energy_act, Lpq, mo_dip = get_pyscf_input_sc(
    mf, with_dip=True, cholesky=True
)

pprpa = ppRPA_Davidson(
    nocc_act,
    mo_energy_act,
    Lpq,
    nroot=2,
    trial="subspace",
    channel="hh",
    mo_dip=mo_dip,
)
pprpa._use_Lov = True
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

dips = []
for i in range(pprpa.nroot):
    xy = pprpa.xy_t[i]
    multi = "t"
    dip = get_dipole_moment(pprpa.nocc, pprpa.mo_dip, pprpa.channel, xy, multi)
    print(f"Dipole moment for state {i}: {dip}")
    dips.append(dip)

if pprpa.nroot == 1:
    quit()

for i in range(1, len(dips)):
    diff = np.array(dips[i]) - np.array(dips[0])
    print(f"Dipole moment difference for state {i} - state 0: {diff}")
    mag = np.linalg.norm(diff)
    print(f"Magnitude: {mag} AU")
