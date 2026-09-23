"""Gamma-point hcore force contracted in density space instead of per atom.

``pprpa_gamma_gpu`` used to call ``krhf.hcore_generator`` once per atom and
trace the resulting AO matrix against the density ``T``: ``natm * 3 * nao^2 *
ngrid`` flops, plus a fresh evaluation of the whole AO grid inside every call.
The matrix is never needed -- the AO indices are contracted on both sides by
the same grid point, so

    Tr[hcore^x_A T] = sum_g vloc_R^x_A(g) rho_T(g)

and by Parseval that is a reduction over G with no FFT per atom.

gpu4pyscf already has that reduction: ``multigrid.eval_vpplocG_SI_gradient``
(and ``eval_nucG_SI_gradient`` all-electron), which ``krhf.grad_elec`` uses on
its ``multigrid_v2`` branch while the default ``KNumInt`` branch still loops
over atoms.  So this module is only the wiring: rho_T(G) from the uniform-grid
numint, that reduction, and ``contract_h1e_dm`` for the AO-derivative half
(``int1e_ipkin`` plus the local PP on the moved basis functions, which was
never per-atom work).  Measured at NV63 ke=600 against the per-atom loop on the
same card: 786.1 s -> 21.8 s, agreeing to 6.1e-15 relative (job 27160009).

Normalisation follows ``multigrid_v2.evaluate_density_on_g_mesh``: the grid
weight ``vol/ngrids/nkpts`` rides along in rho(G) and the SI-gradient helpers
divide it back out by ``cell.vol``.

``PPRPA_HCORE=dense`` falls back to the per-atom generator, which is the
reference tests/test_gpu_hcore_force.py checks against.
"""
from __future__ import annotations

import os
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import UniformGrids, multigrid
from gpu4pyscf.pbc.dft import numint as gnumint
from gpu4pyscf.pbc.grad import krhf as krhf_g
from gpu4pyscf.pbc.grad.rhf import contract_h1e_dm
from gpu4pyscf.lib.cupy_helper import ensure_numpy

from lib_pprpa.pprpa_util import tstamp


def use_density_path():
    """False when ``PPRPA_HCORE=dense`` asks for the per-atom generator."""
    return os.environ.get("PPRPA_HCORE", "").strip().lower() not in ("dense", "per_atom")


def hcore_force(cell, kpts, T, group=None, verbose=True):
    """``sum_atoms einsum('kxij,kji->x', hcore_deriv(ia), T)`` -> (natm, 3) numpy.

    Numerically equal to the per-atom ``krhf.hcore_generator`` loop kept in
    ``pprpa_gamma_gpu._hcore_force``.  Returns ``(de, stats)``.

    ``group`` is accepted so the call site can pass its ``DeviceGroup``, but
    every kernel here is gpu4pyscf's own and runs on the current device; it is
    recorded in ``stats`` and otherwise unused.
    """
    kpts = np.asarray(kpts).reshape(-1, 3)
    if len(kpts) != 1 or abs(kpts).max() > 1e-9:
        raise NotImplementedError("gpu_hcore_force: Gamma point only")
    nkpts = len(kpts)
    mesh = cell.mesh
    ngrids = int(np.prod(mesh))
    Tg = cp.asarray(np.asarray(T, dtype=np.float64))
    if Tg.ndim == 2:
        Tg = Tg[None]                       # (nkpts, nao, nao)

    # rho_T on the uniform grid.  get_rho hands back block_loop's sorted order;
    # the transform needs the natural (x, y, z) order back.
    t0 = time.perf_counter()
    grids = UniformGrids(cell)
    rho = gnumint.KNumInt().get_rho(cell, Tg, grids, kpts)
    rho_r = cp.empty_like(rho)
    rho_r[grids.argsort()] = rho
    rho_g = gtools.fft(rho_r.reshape(1, -1), mesh)[0] * (cell.vol / ngrids / nkpts)
    rho = rho_r = None
    t_rho = time.perf_counter() - t0

    # local PP (or nuclear attraction) moving with the nucleus: gpu4pyscf's
    # G-space reduction, one separable structure factor per atom.
    t0 = time.perf_counter()
    si_grad = (multigrid.eval_vpplocG_SI_gradient if cell._pseudo
               else multigrid.eval_nucG_SI_gradient)
    de = np.asarray(ensure_numpy(si_grad(cell, mesh, rho_g) * nkpts), dtype=np.float64)
    rho_g = None
    t_reduce = time.perf_counter() - t0

    # AO-derivative half.  hermi=0 makes contract_h1e_dm evaluate both of
    # hcore_generator's '-=' lines instead of doubling the first, so a
    # non-symmetric T stays exact.
    t0 = time.perf_counter()
    de -= contract_h1e_dm(cell, krhf_g.get_hcore(cell, kpts), Tg, hermi=0)
    t_h1 = time.perf_counter() - t0

    nslots = getattr(group, "nslots", 1)
    stats = {"label": "hcore_density", "nslots": nslots, "natm": cell.natm,
             "ngrid": ngrids, "rho_seconds": t_rho, "reduce_seconds": t_reduce,
             "h1_seconds": t_h1, "wall_seconds": t_rho + t_reduce + t_h1}
    if verbose:
        print(f"{tstamp()} [gpu_hcore] density-space hcore force: rho {t_rho:.1f} s, "
              f"G-space reduction {t_reduce:.1f} s ({cell.natm} atoms), "
              f"AO-derivative term {t_h1:.1f} s", flush=True)
    return de, stats


__all__ = ["hcore_force", "use_density_path"]
