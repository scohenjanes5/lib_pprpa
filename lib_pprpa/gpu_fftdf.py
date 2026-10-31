'''FFTDF whose Gamma-point SCF exchange comes from the occupied orbitals' factors.

gpu4pyscf's ``fft_jk.get_k_kpts`` forms K[D] for a density tagged with its orbitals from the
nao * nocc codensities phi_p psi_i, complex-to-complex.  With D = C M C^T, C the MOs
(occupied first) and M = diag(mo_occ), the same K is one ``pair_potential_lowrank`` pass
over the real codensities that hold an occupied orbital, nocc (nocc + 1) / 2 + nocc nvir
pairs real-to-complex, and K = S C (C^T K C) C^T S.  Exact on the MO span (all of it unless
the SCF dropped linearly dependent AOs; the SCF only sees that span).  Other densities,
k-points and J go to FFTDF unchanged.
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.df.fft import FFTDF

from lib_pprpa.gpu_pairing_force import pair_exchange, pair_potential_lowrank

__all__ = ['LowRankFFTDF']


class LowRankFFTDF(FFTDF):
    '''FFTDF with the low-rank exchange of an orbital-tagged Gamma-point density
    (``dm.mo_coeff``, ``dm.mo_occ`` as gpu4pyscf's SCF makes them).'''

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True,
               omega=None, exxdiv=None):
        mo_coeff = getattr(dm, 'mo_coeff', None)
        gamma = kpts is None or abs(np.asarray(kpts)).max() < 1e-9
        if (not with_k or mo_coeff is None or kpts_band is not None or not gamma
                or dm.ndim != 3 or dm.shape[0] != 1):
            return super().get_jk(dm, hermi, kpts, kpts_band, with_j, with_k, omega, exxdiv)
        vj = None
        if with_j:
            vj = super().get_jk(dm, hermi, kpts, kpts_band, True, False, omega, exxdiv)[0]
        with self.range_coulomb(omega) as rsh_df:
            # the kernel fft_jk.get_k_kpts uses, cell.omega set for range separation
            coulG = gtools.get_coulG(rsh_df.cell, np.zeros(3), exxdiv, mesh=rsh_df.mesh,
                                     kpts=np.zeros((1, 3)))
        vk = _occupied_exchange(self.cell, self.mesh, cp.asarray(mo_coeff[0]),
                                cp.asarray(dm.mo_occ[0]), coulG)
        return vj, vk[None]


def _occupied_exchange(cell, mesh, mo, occ, coulG):
    '''K[mo diag(occ) mo^T] (nao, nao) CuPy.'''
    order = cp.argsort(occ <= 0, kind='stable')               # occupied columns first
    C, n = mo[:, order], occ[order]
    nocc = int((n > 0).sum())
    M = cp.diag(n)
    W = pair_potential_lowrank(cell, mesh, C.get(), M.get(), coulG=coulG, nlead=nocc)
    KC = cp.asarray(pair_exchange(cell, mesh, W))
    sc = cp.asarray(cell.pbc_intor('int1e_ovlp', hermi=1)) @ C
    return sc @ (C.T @ KC) @ sc.T
