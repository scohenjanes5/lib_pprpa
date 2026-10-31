'''FFTDF whose Gamma-point exchange comes from the factors of the density, in bounded memory.

gpu4pyscf's ``fft_jk.get_k_kpts`` holds the AO values, the AO x density products and the
potential on the whole grid, three (nao, ngrid) arrays, and transforms nao * nocc codensities
complex-to-complex.  Here D = U diag(lam) U^T (the occupied MOs of an orbital-tagged density,
otherwise the eigen-factors of D over an orthonormalised AO basis C) and K C comes from
``gpu_exchange.exchange_ket``: rank x nmo real codensities, the C columns in blocks sized to
the free memory; K = P^T K P with P the S-projector onto the span of C (exact there).  k-points
and J go to FFTDF.
'''

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.df.fft import FFTDF

from lib_pprpa.gpu_exchange import exchange_ket, factor, projector

__all__ = ['LowRankFFTDF']

# Overlap eigenvalues below this fraction of the largest are left out of the orthonormal
# basis of an untagged density (canonical orthogonalisation).
_LINDEP_TOL = 1e-10


class LowRankFFTDF(FFTDF):
    '''FFTDF with the low-rank, memory-bounded exchange of a Gamma-point density.'''

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None, with_j=True, with_k=True,
               omega=None, exxdiv=None):
        gamma = kpts is None or abs(np.asarray(kpts)).max() < 1e-9
        if not with_k or kpts_band is not None or not gamma or dm.ndim != 3 or dm.shape[0] != 1:
            return super().get_jk(dm, hermi, kpts, kpts_band, with_j, with_k, omega, exxdiv)
        vj = None
        if with_j:
            vj = super().get_jk(dm, hermi, kpts, kpts_band, True, False, omega, exxdiv)[0]
        with self.range_coulomb(omega) as rsh_df:
            # the kernel fft_jk.get_k_kpts uses, cell.omega set for range separation
            coulG = gtools.get_coulG(rsh_df.cell, np.zeros(3), exxdiv, mesh=rsh_df.mesh,
                                     kpts=np.zeros((1, 3)))
        cell = self.cell
        ovlp = cp.asarray(cell.pbc_intor('int1e_ovlp', hermi=1))
        mo_coeff = getattr(dm, 'mo_coeff', None)
        if mo_coeff is not None:
            C = cp.asarray(mo_coeff[0])
            occ = cp.asarray(dm.mo_occ[0])
            lam, U = occ[occ > 0], C[:, occ > 0]
            proj = projector(C, ovlp)
        else:
            s, v = cp.linalg.eigh(ovlp)
            keep = s > _LINDEP_TOL * float(s.max())
            C = v[:, keep] / cp.sqrt(s[keep])
            proj = projector(C, ovlp)
            lam, U = factor(proj @ cp.asarray(dm[0]) @ proj.T, C)
        KC = exchange_ket(cell, self.mesh, lam, U, C, coulG, verbose=self.verbose)
        # P^T K P with P = C proj the S-projector onto the span of C
        return vj, (proj.T @ (C.T @ KC) @ proj)[None]
