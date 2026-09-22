"""GPU Gamma-point RKS response  vresp(dm) = J[dm] + fxc . dm  for the CPHF solve.

Why this exists
---------------
The CPHF inside ``make_rdm1_relaxed_rhf_pprpa`` calls the response function
once per iteration (20-40 times per force).  The previous GPU implementation
(``pprpa_gamma_gpu.make_gpu_vresp``) did, per call: one AO pass with
gradients for the XC kernel (recomputing the *ground-state* density and the
kernel every time), a separate FFTDF ``get_j`` that evaluates the full
(nao, ngrid) AO grid at once (90 GB at NV216) and runs two more dense
contractions, all on one GPU.  At NV216 that was ~36 s per iteration and the
CPHF (15 min) had become the largest phase of the force.

Here, for LDA / GGA functionals on the uniform FFT grid:

* the ground-state kernel ``fxc(g)`` is computed once and cached per slot
  (16 doubles per grid point for GGA);
* the grid is split into contiguous ranges over the ``gpu_multi.DeviceGroup``
  slots, each slot streaming its range in chunks whose AO values never leave
  the device;
* the Coulomb potential of the perturbing density is folded into the XC
  weights, so the second pass builds ``J + fxc`` with one GEMM instead of
  three dense contractions plus a second AO grid.

Per call: pass 1 (AO values, one GEMM) gives the density for J; its potential
comes from the R2C Coulomb chain in ``gpu_coulomb``; pass 2 (AO values and
gradients for GGA, two GEMMs) forms the response matrix.  Conventions follow
``grad_utils_gpu_pbc.nr_rks_fxc`` (hermi=1, symmetrised) and
``fft_jk.get_j_kpts``; validated against the dense implementation in
tests/test_gpu_response.py.  Hybrids add ``-0.5 * hyb * K[dm]`` through the
dense ``kmf.get_k``, as before.
"""
from __future__ import annotations

import os
import time

import numpy as np
import cupy as cp
from gpu4pyscf.pbc import tools as gtools
from gpu4pyscf.pbc.dft import numint as gnumint
from gpu4pyscf.lib.cupy_helper import contract

from lib_pprpa.gpu_coulomb import coulomb_potential, half_kernel
from lib_pprpa.gpu_multi import _free_bytes, _is_oom, _reclaim_gpu, default_group
from lib_pprpa.pprpa_util import tstamp

_KPTS0 = np.zeros((1, 3))
LAST_TELEMETRY = {}


def get_last_telemetry():
    return dict(LAST_TELEMETRY)


def _log(msg):
    print(f"{tstamp()} [gpu_response] {msg}", flush=True)


def _grid_chunk(nao, ncomp, ngrid, frac=0.3, floor=4096):
    """Grid points per chunk: AO block (ncomp comps) + its evaluation transient
    + c0 + aow + one temporary, all (g, nao) doubles."""
    per_point = 8 * int(nao) * (2 * int(ncomp) + 3)
    budget = int(_free_bytes() * frac)
    return int(max(min(floor, ngrid), min(ngrid, budget // max(1, per_point))))


class GammaResponse:
    """Callable ``vresp(dm) -> numpy (nao, nao)`` for Gamma RKS with an LDA/GGA
    functional; ``dm`` is a symmetric numpy/cupy density matrix (hermi=1)."""

    def __init__(self, cell, mf, group=None, mesh=None, gchunk=None, verbose=True):
        from gpu4pyscf.pbc import dft as gdft
        self.cell = cell
        self.group = group or default_group()
        self.mesh = np.asarray(cell.mesh if mesh is None else mesh, dtype=int)
        self.ngrid = int(np.prod(self.mesh))
        self.nao = int(cell.nao)
        self.verbose = verbose
        self.xc = mf.xc
        kmf = gdft.KRKS(cell, kpts=_KPTS0, xc=mf.xc)
        kmf.exxdiv = None
        self.kmf = kmf
        self.ni = kmf._numint
        self.xctype = self.ni._xc_type(mf.xc)
        if self.xctype not in ("LDA", "GGA"):
            raise NotImplementedError(f"GammaResponse: {mf.xc} ({self.xctype}); LDA/GGA only")
        self.gga = self.xctype == "GGA"
        self.ncomp = 4 if self.gga else 1
        self.omega, self.alpha, self.hyb = self.ni.rsh_and_hybrid_coeff(mf.xc)
        self.weight = cell.vol / self.ngrid
        self.coords = cell.gen_uniform_grids(self.mesh)
        self._gchunk = gchunk
        self.stats = {"calls": 0, "pass1_seconds": 0.0, "coulomb_seconds": 0.0,
                      "pass2_seconds": 0.0, "seconds": 0.0, "init_seconds": 0.0}
        n = self.group.nslots
        self.ranges = [((self.ngrid * r) // n, (self.ngrid * (r + 1)) // n) for r in range(n)]
        dm0 = np.asarray(mf.make_rdm1(), dtype=np.float64)
        if dm0.ndim == 3:
            dm0 = dm0[0]
        t0 = time.perf_counter()
        self._init_slots(dm0)
        self.stats["init_seconds"] = time.perf_counter() - t0
        if verbose:
            _log(f"{self.xctype} response on {n} slot(s): ngrid={self.ngrid} nao={self.nao} "
                 f"fxc cached in {self.stats['init_seconds']:.1f} s")

    # ------------------------------------------------------------------ setup
    def _init_slots(self, dm0):
        coulG = gtools.get_coulG(self.cell, k=np.zeros(3), exx=None, mesh=self.mesh)
        self._w_half = half_kernel(cp.asarray(coulG), self.mesh)

        def _setup(ctx):
            st = ctx.state
            g0, g1 = self.ranges[ctx.rank]
            st["rng"] = (g0, g1)
            st["opt0"] = gnumint._GTOvalOpt(self.cell, _KPTS0, deriv=0)
            st["opt1"] = gnumint._GTOvalOpt(self.cell, _KPTS0, deriv=1) if self.gga else st["opt0"]
            st["gchunk"] = (int(self._gchunk) if self._gchunk
                            else _grid_chunk(self.nao, self.ncomp, g1 - g0))
            dmg = cp.asarray(dm0)
            nvar = 4 if self.gga else 1
            fxc = cp.empty((nvar, nvar, g1 - g0), dtype=cp.float64)
            for c0, c1 in self._chunks(ctx):
                ao = self._ao(ctx, c0, c1, self.gga)
                rho0 = self._rho(ao, dmg)
                fxc[:, :, c0 - g0:c1 - g0] = self.ni.eval_xc_eff(
                    self.xc, rho0, deriv=2, xctype=self.xctype)[2]
                ao = rho0 = None
            st["fxc"] = fxc
            _reclaim_gpu()
        self.group.each(_setup)

    def _chunks(self, ctx):
        g0, g1 = ctx.state["rng"]
        step = int(ctx.state["gchunk"])
        return [(a, min(g1, a + step)) for a in range(g0, g1, step)]

    def _ao(self, ctx, c0, c1, deriv1):
        opt = ctx.state["opt1"] if deriv1 else ctx.state["opt0"]
        return gnumint.eval_ao_kpts(self.cell, self.coords[c0:c1], kpts=_KPTS0,
                                    deriv=1 if deriv1 else 0, opt=opt)[0]

    def _rho(self, ao, dmg):
        """Density (and gradient for GGA) of a symmetric dm on the chunk:
        mirrors KNumInt.eval_rho(hermi=1) -- rho[0] = c0.ao0, rho[1:] = 2 c0.ao_x."""
        if self.gga:
            c0 = ao[0] @ dmg
            rho = cp.empty((4, ao.shape[1]), dtype=cp.float64)
            rho[0] = contract("gi,gi->g", c0, ao[0])
            for x in range(1, 4):
                rho[x] = 2.0 * contract("gi,gi->g", c0, ao[x])
            return rho
        c0 = ao @ dmg
        return contract("gi,gi->g", c0, ao)[None]

    # ------------------------------------------------------------------- call
    def __call__(self, dm):
        dm_h = np.ascontiguousarray(cp.asnumpy(dm) if isinstance(dm, cp.ndarray) else dm,
                                    dtype=np.float64)
        if dm_h.ndim == 3:
            dm_h = dm_h[0]
        t0 = time.perf_counter()

        # pass 1: density of dm on the grid (values only) for the Coulomb term
        def _pass1(ctx):
            st = ctx.state
            g0, g1 = st["rng"]
            dmg = cp.asarray(dm_h)
            out = cp.empty(g1 - g0, dtype=cp.float64)
            for c0, c1 in self._chunks(ctx):
                ao = self._ao(ctx, c0, c1, False)
                cdm = ao @ dmg
                out[c0 - g0:c1 - g0] = contract("gi,gi->g", cdm, ao)
                ao = cdm = None
            return cp.asnumpy(out)
        rho_h = np.concatenate(self._run(_pass1))
        t1 = time.perf_counter()

        # Coulomb potential of that density: vR = ifft(coulG * fft(rho)), as fft_jk.get_j_kpts
        vR_h = cp.asnumpy(coulomb_potential(cp.asarray(rho_h)[None], self.mesh, self._w_half,
                                            rfft=True)[0])
        t2 = time.perf_counter()

        # pass 2: J + fxc response matrix, one GEMM per chunk
        weight = self.weight
        gga = self.gga

        def _pass2(ctx):
            st = ctx.state
            g0, g1 = st["rng"]
            dmg = cp.asarray(dm_h)
            vR = cp.asarray(vR_h[g0:g1])
            fxc = st["fxc"]
            vmat = cp.zeros((self.nao, self.nao), dtype=cp.float64)
            for c0, c1 in self._chunks(ctx):
                ao = self._ao(ctx, c0, c1, gga)
                rho1 = self._rho(ao, dmg)
                wv = cp.einsum("xg,xyg->yg", rho1, fxc[:, :, c0 - g0:c1 - g0]) * weight
                if gga:
                    wv[0] *= 0.5
                    wv[0] += 0.5 * weight * vR[c0 - g0:c1 - g0]     # J folded in (h.c. doubles it)
                    aow = ao[0] * wv[0][:, None]
                    for x in range(1, 4):
                        aow += ao[x] * wv[x][:, None]
                    vmat += ao[0].T @ aow
                else:
                    wv0 = wv[0] + weight * vR[c0 - g0:c1 - g0]
                    vmat += ao.T @ (ao * wv0[:, None])
                ao = rho1 = wv = aow = None
            return cp.asnumpy(vmat)
        v = np.sum(self._run(_pass2), axis=0)
        if gga:
            v = v + v.T
        t3 = time.perf_counter()

        if abs(self.hyb) > 1e-12:
            dmg = cp.asarray(dm_h)
            v = v - 0.5 * self.hyb * cp.asnumpy(
                self.kmf.get_k(self.cell, dmg[None], hermi=1, kpts=_KPTS0)[0])

        st = self.stats
        st["calls"] += 1
        st["pass1_seconds"] += t1 - t0
        st["coulomb_seconds"] += t2 - t1
        st["pass2_seconds"] += t3 - t2
        st["seconds"] += time.perf_counter() - t0
        return v

    def _run(self, fn):
        """``fn(ctx)`` on every slot, halving that slot's grid chunk on an OOM."""
        def _guarded(ctx):
            while True:
                try:
                    return fn(ctx)
                except Exception as exc:  # noqa: BLE001 - classified below
                    if not _is_oom(exc) or int(ctx.state["gchunk"]) <= 1:
                        raise
                    _reclaim_gpu()
                    ctx.state["gchunk"] = max(1, int(ctx.state["gchunk"]) // 2)
                    ctx.log(f"gpu_response: {type(exc).__name__}; retrying with "
                            f"gchunk={ctx.state['gchunk']}")
        return self.group.each(_guarded)

    # -------------------------------------------------------------- telemetry
    def summary(self):
        st = self.stats
        LAST_TELEMETRY.clear()
        LAST_TELEMETRY.update(st, nslots=self.group.nslots, xctype=self.xctype)
        if not st["calls"]:
            return "gpu_response: no calls"
        n = st["calls"]
        return (f"gpu_response: {n} calls, {st['seconds']:.1f} s total "
                f"({st['seconds']/n:.2f} s/call: pass1 {st['pass1_seconds']/n:.2f}, "
                f"coulomb {st['coulomb_seconds']/n:.2f}, pass2 {st['pass2_seconds']/n:.2f}); "
                f"fxc cache {st['init_seconds']:.1f} s; {self.group.nslots} slot(s)")

    def release(self):
        self.group.free(["fxc", "opt0", "opt1", "rng", "gchunk"])
        self._w_half = None


def make_gpu_vresp_grouped(cell, mf, group=None, verbose=True):
    """``GammaResponse`` when the functional allows it, else ``None`` (caller
    falls back to the dense response).  ``PPRPA_RESPONSE=dense`` forces the
    fallback."""
    if os.environ.get("PPRPA_RESPONSE", "").strip().lower() == "dense":
        return None
    try:
        return GammaResponse(cell, mf, group=group, verbose=verbose)
    except NotImplementedError as exc:
        if verbose:
            _log(f"{exc}; using the dense response")
        return None
