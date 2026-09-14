"""Shared VRAM accounting for GPU ao2mo and Davidson ``use_eri``.

The resident path keeps all three active-space ERI tensors on the GPU:

    vvvv : nv^4 * 8 B
    oooo : no^4 * 8 B
    oovv : no^2 * nv^2 * 8 B

``fits_resident`` is the same 75% of device memory rule that
``gpu_ao2mo.gpu_ao2mo_blocks`` uses to decide host staging.
"""
from __future__ import annotations

RESIDENT_VRAM_FRAC = 0.75

# cuFFT rejects a batched plan whose total element count exceeds 2**31-1 when a
# transform dimension takes the Bluestein (chirp-z) path, i.e. has a prime
# factor above 127 that the direct radix kernels cannot handle (e.g. the 151^3
# mesh of the 63-atom cell at ke=600 Ha): CUFFT_INVALID_SIZE, not an OOM.
# Direct-path meshes (107, 128, 159 = 3*53) ran 1.6 * 2**31-element batches on a
# B200 (probe job 25821985); they get a relaxed cap inside that tested range so
# whole-MO-row strips are not needlessly halved (159^3: 534 -> 300 pairs cost
# the 216-atom ao2mo ~1.3 h per force).
CUFFT_MAX_PLAN_ELEMENTS = 2 ** 31 - 1
CUFFT_MAX_PLAN_ELEMENTS_DIRECT = int(1.5 * 2 ** 31)
CUFFT_MAX_DIRECT_PRIME = 127


def _largest_prime_factor(n):
    n = int(n)
    largest = 1
    p = 2
    while p * p <= n:
        while n % p == 0:
            largest = p
            n //= p
        p += 1
    return max(largest, n) if n > 1 else largest


def needs_bluestein(mesh, max_direct_prime=CUFFT_MAX_DIRECT_PRIME):
    """True if any FFT dimension has a prime factor cuFFT cannot handle directly."""
    return any(_largest_prime_factor(n) > max_direct_prime for n in mesh)


def max_fft_batch(ngrid, mesh=None, limit=None):
    """Largest number of ``ngrid``-point transforms one cuFFT plan may batch.

    Without ``mesh`` the strict 2^31 cap applies; with a mesh whose dimensions
    all avoid the Bluestein path the relaxed (tested) cap is used.
    """
    if limit is None:
        limit = CUFFT_MAX_PLAN_ELEMENTS
        if mesh is not None and not needs_bluestein(mesh):
            limit = CUFFT_MAX_PLAN_ELEMENTS_DIRECT
    return max(1, int(limit) // int(ngrid))


def eri_bytes(nocc, nvir):
    """Host/device size of vvvv + oooo + oovv as float64, in bytes."""
    no = int(nocc)
    nv = int(nvir)
    return (nv ** 4 + no ** 4 + (no ** 2) * (nv ** 2)) * 8


def total_vram_bytes():
    """Driver-reported total device memory."""
    import cupy as cp
    _free, total = cp.cuda.runtime.memGetInfo()
    return int(total)


def fits_resident(nocc, nvir, extra_bytes=0, total_bytes=None,
                  frac=RESIDENT_VRAM_FRAC):
    """True if the three ERI tensors plus ``extra_bytes`` fit in ``frac`` of VRAM.

    Pass ``total_bytes`` to avoid a CUDA query (tests, or after a cached
    ``memGetInfo``).  If omitted, the live device total is used.
    """
    if total_bytes is None:
        total_bytes = total_vram_bytes()
    budget = int(frac * int(total_bytes))
    return eri_bytes(nocc, nvir) + int(extra_bytes) <= budget
