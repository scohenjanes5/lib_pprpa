'''Gamma-point KRKS reference of the GPU pp-RPA, and the DFT gradient of that reference.

``make_krks`` is gpu4pyscf's KRKS at Gamma with ``exxdiv`` None; a hybrid's exact exchange is
``gpu_fftdf.LowRankFFTDF``'s adaptively compressed exchange.  ``run_scf`` converges it and,
after an ACE SCF, puts in the orbitals of the full exchange (``full_exchange_orbitals``): the
ACE Fock is exact on the occupied orbitals only, and the pp-RPA needs the virtual ones.
``dft_gradient`` is the nuclear gradient of the converged reference: gpu4pyscf's multigrid
KRKS gradient of the semilocal part at these orbitals (the overlap term with their energies)
plus, for a global hybrid, the exact-exchange force of the occupied density
(``gpu_exchange.reference_exchange_force``).  Closed shell; no gradient for range-separated hybrids.
'''

import re

import numpy as np
import cupy as cp
from pyscf.dft import libxc
from pyscf.lib import logger
from gpu4pyscf.pbc import dft as gdft

from lib_pprpa.gpu_coulomb import exchange_coulG
from lib_pprpa.gpu_exchange import reference_exchange_force
from lib_pprpa.gpu_fftdf import LowRankFFTDF, full_exchange_orbitals
from lib_pprpa.gpu_mem import pooled

__all__ = ['make_krks', 'run_scf', 'semilocal_xc', 'dft_gradient']

_GAMMA = np.zeros((1, 3))


def make_krks(cell, xc, conv_tol=1e-9):
    '''gpu4pyscf KRKS at Gamma (``exxdiv`` None); a hybrid ``xc`` gets the ACE exchange.'''
    kmf = gdft.KRKS(cell, kpts=_GAMMA, xc=xc)
    kmf.exxdiv = None
    kmf.conv_tol = conv_tol
    if libxc.is_hybrid_xc(xc):
        kmf.with_df = LowRankFFTDF(cell, _GAMMA)
        kmf.with_df.ace = True
    return kmf


def run_scf(kmf, dm0=None):
    '''Converge ``kmf`` from ``dm0``, its grid blocks from cupy's pool; after an ACE SCF the
    orbitals and energies become those of the full exchange.  Raises RuntimeError when the
    SCF does not converge.'''
    with pooled():
        kmf.kernel(dm0)
        if not kmf.converged:
            raise RuntimeError(f'KRKS SCF did not converge (E = {kmf.e_tot} Ha)')
        if getattr(kmf.with_df, 'ace', False):
            log = logger.new_logger(kmf)
            t0 = log.init_timer()
            full_exchange_orbitals(kmf)
            log.timer('full-exchange orbitals', *t0)
    return kmf


def semilocal_xc(xc):
    '''The semilocal part of a global hybrid ``xc``, its HF terms removed
    ("0.18*HF+0.82*PBE,PBE" -> "0.82*PBE,PBE").  ValueError for a range-separated hybrid or an
    ``xc`` without HF terms.'''
    if libxc.rsh_coeff(xc)[0] != 0:
        raise ValueError(f'{xc}: range-separated hybrid')
    xpart, sep, cpart = xc.partition(',')
    terms = [t for t in re.split(r'(?=[+-])', xpart.replace(' ', '')) if t]
    kept = [t for t in terms if not re.fullmatch(r'[+-]?([0-9.eE+-]*\*)?HF', t, re.I)]
    sl = ''.join(kept).lstrip('+') + sep + cpart.replace(' ', '')
    if len(kept) == len(terms) or libxc.is_hybrid_xc(sl):
        raise ValueError(f'{xc}: no HF terms to separate from a semilocal part')
    return sl


def dft_gradient(kmf, verbose=None):
    '''dE/dR (NumPy, (natm, 3), Ha/Bohr) of the converged closed-shell Gamma KRKS ``kmf``.'''
    cell, xc = kmf.cell, kmf.xc
    log = logger.new_logger(kmf, verbose)
    t0 = log.init_timer()
    hybrid = libxc.is_hybrid_xc(xc)
    sl = semilocal_xc(xc) if hybrid else xc
    ms = gdft.KRKS(cell, kpts=_GAMMA, xc=sl).multigrid_numint()
    ms.exxdiv = None
    ms.mo_coeff, ms.mo_energy, ms.mo_occ = kmf.mo_coeff, kmf.mo_energy, kmf.mo_occ
    ms.converged = True
    de = np.asarray(ms.Gradients().kernel())
    t0 = log.timer(f'semilocal ({sl}) gradient', *t0)
    if hybrid:
        omega, alpha, hyb = kmf._numint.rsh_and_hybrid_coeff(xc)
        coulG = exchange_coulG(cell, cell.mesh, omega, alpha, hyb, None)
        de += reference_exchange_force(cell, cell.mesh, cp.asnumpy(kmf.mo_coeff[0]),
                                       cp.asnumpy(kmf.mo_occ[0]), coulG)
        log.timer('exact-exchange gradient', *t0)
    return de
