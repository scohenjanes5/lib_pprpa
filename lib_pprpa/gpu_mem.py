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
