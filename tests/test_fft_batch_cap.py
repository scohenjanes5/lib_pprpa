"""cuFFT batched-plan size cap shared by gpu_ao2mo and gpu_fft_k.

Background: on a B200 the ao2mo strip planner chose pair_blk=1200 for the
151^3 mesh (ke=600 Ha, 63-atom NV cell) and cuFFT failed with
CUFFT_INVALID_SIZE, because 1200 * 151^3 > 2^31 and 151 (prime > 127) uses the
Bluestein path.  The H100 run of the same job used pair_blk=300 and passed.
The pure-Python parts run anywhere; the planner checks need cupy + gpu4pyscf.
"""
from __future__ import annotations

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from lib_pprpa.gpu_mem import CUFFT_MAX_PLAN_ELEMENTS, max_fft_batch


def _need_gpu_modules():
    try:
        import cupy  # noqa: F401
        from lib_pprpa import gpu_ao2mo  # noqa: F401
    except Exception:
        if pytest is None:
            raise RuntimeError("cupy + gpu4pyscf required") from None
        pytest.skip("cupy + gpu4pyscf required")


def test_cap_mesh_151():
    ng = 151 ** 3
    b = max_fft_batch(ng)
    assert b == 623
    assert b * ng <= CUFFT_MAX_PLAN_ELEMENTS < (b + 1) * ng
    assert 1200 * ng > CUFFT_MAX_PLAN_ELEMENTS      # the failing B200 strip


def test_cap_mesh_159_and_small():
    assert max_fft_batch(159 ** 3) == 534               # strict when the mesh is unknown
    assert max_fft_batch(107 ** 3) == 1752
    assert max_fft_batch(10 ** 12) == 1              # never below one transform


def test_bluestein_detection_and_mesh_aware_cap():
    from lib_pprpa.gpu_mem import (CUFFT_MAX_PLAN_ELEMENTS_DIRECT, _largest_prime_factor,
                                   needs_bluestein)
    assert _largest_prime_factor(151) == 151 and _largest_prime_factor(159) == 53
    assert _largest_prime_factor(128) == 2 and _largest_prime_factor(107) == 107
    assert needs_bluestein([151, 151, 151])
    assert needs_bluestein([128, 128, 151])            # one bad dimension is enough
    assert not needs_bluestein([159, 159, 159])
    assert not needs_bluestein([107, 107, 107])        # 107 <= 127: direct kernels
    assert not needs_bluestein([128, 128, 128])
    # direct-path mesh: relaxed cap inside the range probed on the B200 (1.6 * 2^31)
    assert max_fft_batch(159 ** 3, mesh=[159, 159, 159]) == 801
    assert 801 * 159 ** 3 <= CUFFT_MAX_PLAN_ELEMENTS_DIRECT < 1.6 * 2 ** 31
    assert max_fft_batch(107 ** 3, mesh=[107, 107, 107]) == 2629
    # Bluestein mesh: strict cap regardless
    assert max_fft_batch(151 ** 3, mesh=[151, 151, 151]) == 623


def test_plan_strips_caps_the_fft_batch_not_the_gemm_strip():
    _need_gpu_modules()
    from lib_pprpa.gpu_ao2mo import _plan_strips, _estimate_pair_blk
    ng = 151 ** 3
    free = 150e9
    # The GEMM strip is decoupled from the cuFFT plan: an explicit 1200-pair
    # request stands, while the FFT sub-batch that fills its potential stays
    # under the plan limit (Bluestein mesh: strict 2^31 -> 623 transforms).
    blk, fblk = _plan_strips(npair=90000, ngrid=ng, nB=300, pair_blk=1200, mesh=[151] * 3,
                             free=free)
    assert blk == 1200 and fblk <= 623 and fblk * ng <= CUFFT_MAX_PLAN_ELEMENTS
    assert _estimate_pair_blk(npair=90000, ngrid=ng, nB=300, pair_blk=1200, mesh=[151] * 3,
                              free=free) == 1200
    # a forced FFT batch above the cap is clamped, and never exceeds the strip
    _blk, fblk = _plan_strips(npair=90000, ngrid=ng, nB=300, pair_blk=300, fft_blk=5000,
                              mesh=[151] * 3, free=free)
    assert fblk == 300
    # planner: full-index strips are whole nB rows; compact strips need no alignment
    blk, fblk = _plan_strips(npair=90000, ngrid=159 ** 3, nB=300, mesh=[159] * 3, free=free)
    assert blk % 300 == 0 and blk >= 300 and fblk >= 1
    blkc, _f = _plan_strips(npair=45150, ngrid=159 ** 3, nB=300, mesh=[159] * 3, free=free,
                            compact=True)
    assert blkc >= 300
    # the FFT sub-batch is a minor share of the budget, so the GEMM strip is far
    # wider than the old 64 B/pair-gridpoint plan allowed (600 at ~170 GB free)
    assert blk >= 1200


def test_is_oom_treats_cufft_size_errors_as_retryable():
    _need_gpu_modules()
    from lib_pprpa.gpu_ao2mo import _is_oom
    CuFFTError = type("CuFFTError", (Exception,), {})
    assert _is_oom(CuFFTError("CUFFT_INVALID_SIZE"))
    assert _is_oom(CuFFTError("CUFFT_ALLOC_FAILED"))
    assert not _is_oom(CuFFTError("CUFFT_INTERNAL_ERROR"))
    assert not _is_oom(ValueError("shape mismatch"))


if __name__ == "__main__":
    test_cap_mesh_151()
    test_cap_mesh_159_and_small()
    test_bluestein_detection_and_mesh_aware_cap()
    print("OK  pure-python caps")
    test_plan_strips_caps_the_fft_batch_not_the_gemm_strip()
    test_is_oom_treats_cufft_size_errors_as_retryable()
    print("OK  planner + retry classification")
