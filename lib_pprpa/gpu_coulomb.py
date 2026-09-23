"""Shared GPU pieces of the batched-FFT Coulomb kernels.

Three kernels in this package do the same thing to a batch of real codensity
functions ``rho_P(g) = A[a_P](g) * B[b_P](g)``:

    vR = ifft(w(G) * fft(rho))          (w = (vol/ngrid) * 4 pi / |G|^2)

-- the ao2mo pair strips (``gpu_ao2mo``), the low-rank exchange
(``gpu_fft_k``) and the pairing force (``gpu_pairing_force``).  What differs
is only what is contracted with ``vR`` afterwards and how much memory that
contraction holds, so this module owns the common part -- the fused codensity
kernel, the real-to-complex potential chain and the per-point memory cost of
one FFT batch -- and every kernel keeps its own planner for the rest.  An
earlier design shared one planner and, with it, one byte-per-point guess and
one coupling of "contraction block = FFT batch"; that held the ao2mo GEMM at
a third of the card's rate (see docs/RESULTS.md).

The R2C chain must use the *symmetrised* half-mesh kernel
(``pair_layout.symmetric_half_kernel``): pyscf's convention
``ifft(fft(rho) * w).real`` only ever sees the even part of ``w``, and on the
Nyquist planes of an even mesh dimension in a non-orthogonal cell ``w`` is not
even.  With that kernel R2C reproduces the C2C result to rounding on every
mesh (tests/test_pair_layout.py, and the ao2mo diamond self-test).
"""
from __future__ import annotations

import os

import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from gpu4pyscf.pbc import tools as gtools

from lib_pprpa.gpu_mem import max_fft_batch
from lib_pprpa.pair_layout import symmetric_half_kernel

# Bytes per (function, grid point) alive while one FFT batch is transformed.
# C2C: real rho (8) + complex cast (16) + vG (16) + complex vR (16).
# R2C: rho (8) + half spectrum (8, multiplied in place) + real vR (8) + cuFFT
# work area for the batched R2C / C2R plans (~16).  24 for R2C was measured too
# tight on a B200 (OOM + retry).
FFT_BYTES_C2C = 56
FFT_BYTES_R2C = 40
# Below this many transforms per batch the per-call overhead dominates.
FFT_BLK_FLOOR = 16


def env_flag(name, default=False):
    v = os.environ.get(name)
    if v is None or not v.strip():
        return default
    return v.strip().lower() in ("1", "true", "yes")


def use_rfft(rfft=None):
    """Resolve an ``rfft`` argument: None -> ``GPU_FFT_RFFT`` (default on)."""
    if rfft is None:
        return env_flag("GPU_FFT_RFFT", True)
    return bool(rfft)


def fft_bytes_per_point(rfft):
    return FFT_BYTES_R2C if rfft else FFT_BYTES_C2C


# rho[P, g] = A[aidx[P], g] * B[bidx[P], g] in one pass: two reads and one write
# per element and no gathered temporaries (take + in-place multiply needs a
# second (blk, ngrid) array and twice the traffic).
CODENSITY_KERNEL = cp.ElementwiseKernel(
    "raw float64 A, raw float64 B, raw int32 aidx, raw int32 bidx, int64 ngrid, int64 P0",
    "float64 rho",
    """
    const long long row = i / ngrid;
    const long long g = i - row * ngrid;
    const long long P = P0 + row;
    rho = A[(long long)aidx[P] * ngrid + g] * B[(long long)bidx[P] * ngrid + g];
    """,
    "pprpa_codensity")


def codensity(A, B, aidx, bidx, P0, P1, out=None):
    """Codensities of pairs [P0, P1) -> (P1-P0, ngrid); ``A``, ``B`` are
    C-contiguous (n, ngrid) grids, ``aidx`` / ``bidx`` int32 device arrays
    mapping a pair to its rows.  Written into ``out`` when given."""
    ngrid = A.shape[1]
    if out is None:
        out = cp.empty((P1 - P0, ngrid), dtype=cp.float64)
    CODENSITY_KERNEL(A, B, aidx, bidx, ngrid, P0, out)
    return out


def half_kernel(w_flat, mesh):
    """The symmetrised kernel on the R2C half mesh, (nx, ny, nz//2 + 1), cupy."""
    return cp.ascontiguousarray(symmetric_half_kernel(cp.asarray(w_flat), mesh))


def coulomb_potential(rho, mesh, w, rfft=True):
    """``ifft(fft(rho) * w)`` for a batch of real functions, (f, ngrid) -> (f, ngrid) real.

    ``rfft=False``: gpu4pyscf's complex chain with the flat kernel ``w``;
    ``rfft=True``: rfftn, in-place product with the half-mesh kernel ``w``
    (``half_kernel``), irfftn straight back to real -- half the spectral
    traffic, no complex cast, no ``.real`` copy.
    """
    if not rfft:
        return gtools.ifft(gtools.fft(rho, mesh) * w, mesh).real
    f = rho.shape[0]
    nx, ny, nz = (int(m) for m in mesh)
    vG = cufft.rfftn(rho.reshape(f, nx, ny, nz), axes=(1, 2, 3))
    vG *= w
    vR = cufft.irfftn(vG, s=(nx, ny, nz), axes=(1, 2, 3), overwrite_x=True)
    return vR.reshape(f, -1)


def plan_fft_batch(budget, ngrid, mesh, rfft, nmax, floor=FFT_BLK_FLOOR):
    """Functions per FFT batch from a byte ``budget``: the per-point cost of
    the chain, the cuFFT batched-plan limit for this mesh, and ``nmax``."""
    per = fft_bytes_per_point(rfft) * int(ngrid)
    f = int(max(0, budget)) // max(1, per)
    f = max(floor, f)
    return int(max(1, min(f, max_fft_batch(ngrid, mesh=mesh), nmax)))


def fft_batch_bytes(f, ngrid, rfft):
    return int(f) * fft_bytes_per_point(rfft) * int(ngrid)


def balanced_chunk(n, cap):
    """Chunk size that splits ``n`` into ceil(n/cap) equal parts: never above
    ``cap``, same chunk count, no runt (300 under 269 -> 150 + 150)."""
    cap = max(1, min(int(n), int(cap)))
    nchunk = -(-int(n) // cap)
    return -(-int(n) // nchunk)


__all__ = ["FFT_BYTES_C2C", "FFT_BYTES_R2C", "FFT_BLK_FLOOR", "CODENSITY_KERNEL",
           "codensity", "half_kernel", "coulomb_potential", "plan_fft_batch",
           "fft_batch_bytes", "fft_bytes_per_point", "balanced_chunk", "use_rfft",
           "env_flag", "symmetric_half_kernel", "np"]
