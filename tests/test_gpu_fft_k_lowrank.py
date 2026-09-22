"""Low-rank FFT exchange (lib_pprpa.gpu_fft_k) vs dense gpu4pyscf fft_jk.get_k.

Run directly on a GPU node (``python tests/test_gpu_fft_k_lowrank.py``) or via
pytest.  Checks K[L R^T] for symmetric, antisymmetric and general low-rank
densities, both with the automatic block plan and with tiny forced blocks so the
row/rank chunking paths are exercised.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def _need_cupy():
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDevice()
    except Exception:
        if pytest is None:
            raise RuntimeError("CUDA GPU required for this test") from None
        pytest.skip("CUDA GPU required")


def _two_slots():
    """Two real devices when the allocation has them, else two virtual slots on GPU 0."""
    import cupy as _cp
    return [0, 1] if _cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]


def _diamond_cell(basis="gth-dzvp", ke=120.0, nrep=1):
    from pyscf.pbc import gto
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.0, 0.0, 0.0]], ["C", [a0 / 4 + 0.03, a0 / 4, a0 / 4 - 0.02]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = basis
    cell.pseudo = "gth-pade"
    cell.ke_cutoff = ke
    cell.verbose = 0
    cell.build()
    if nrep > 1:
        from pyscf.pbc.tools import super_cell
        cell = super_cell(cell, [nrep, 1, 1])
        cell.verbose = 0
        cell.build()
    return cell


def _dense_gpu_k(cell, dms):
    import cupy as cp
    from gpu4pyscf.pbc.df.fft import FFTDF
    from gpu4pyscf.pbc.df import fft_jk
    fdf = FFTDF(cell, np.zeros((1, 3)))
    K = fft_jk.get_k(fdf, cp.asarray(dms), hermi=0, kpt=np.zeros(3), exxdiv=None)
    return cp.asnumpy(K)


def _factors(nao, rank, rng):
    C = rng.standard_normal((nao, rank))
    x = rng.standard_normal((rank, rank))
    sym = (C @ (x + x.T) * 0.5, C)                 # symmetric core
    anti = (C @ (x - x.T) * 0.5, C)                # antisymmetric core (triplet X)
    gen = (rng.standard_normal((nao, rank)), rng.standard_normal((nao, rank)))
    return [sym, anti, gen]


def _run_case(cell, rank, row_blk=None, rank_blk=None, seed=0):
    from lib_pprpa.gpu_fft_k import get_k_lowrank, get_last_telemetry
    rng = np.random.default_rng(seed)
    nao = cell.nao
    pairs = _factors(nao, rank, rng)
    dms = np.stack([L @ R.T for L, R in pairs])
    K_ref = _dense_gpu_k(cell, dms)
    K_lr = get_k_lowrank(cell, cell.mesh, pairs, row_blk=row_blk, rank_blk=rank_blk,
                         verbose=False)
    err = np.abs(K_lr - K_ref).max(axis=(1, 2))
    scale = np.abs(K_ref).max(axis=(1, 2))
    # single-pair call path
    K_one = get_k_lowrank(cell, cell.mesh, pairs[1], row_blk=row_blk,
                          rank_blk=rank_blk, verbose=False)
    err_one = np.abs(K_one - K_ref[1]).max()
    return err, scale, err_one, get_last_telemetry()


def test_lowrank_matches_dense_auto_blocks():
    _need_cupy()
    cell = _diamond_cell()
    err, scale, err_one, tel = _run_case(cell, rank=5)
    assert np.all(err < 1e-9 * np.maximum(scale, 1.0)), (err, scale)
    assert err_one < 1e-9 * max(scale[1], 1.0)
    assert tel["nset"] == 1


def test_lowrank_matches_dense_forced_small_blocks():
    _need_cupy()
    cell = _diamond_cell()
    err, scale, err_one, _ = _run_case(cell, rank=7, row_blk=3, rank_blk=2, seed=1)
    assert np.all(err < 1e-9 * np.maximum(scale, 1.0)), (err, scale)
    assert err_one < 1e-9 * max(scale[1], 1.0)


def test_lowrank_rank_exceeds_nothing_supercell():
    """2x supercell, rank close to nao, symmetric + antisymmetric + general."""
    _need_cupy()
    cell = _diamond_cell(basis="gth-szv", ke=100.0, nrep=2)
    err, scale, err_one, _ = _run_case(cell, rank=cell.nao - 2, row_blk=None,
                                       rank_blk=4, seed=2)
    assert np.all(err < 1e-9 * np.maximum(scale, 1.0)), (err, scale)
    assert err_one < 1e-9 * max(scale[1], 1.0)


def test_virtual_two_slots_match_one_slot():
    """DeviceGroup(_two_slots()) on one GPU (threads, replication, row tasks) vs [0]."""
    _need_cupy()
    from lib_pprpa.gpu_fft_k import get_k_lowrank, get_last_telemetry
    from lib_pprpa.gpu_multi import DeviceGroup
    cell = _diamond_cell()
    rng = np.random.default_rng(4)
    pairs = _factors(cell.nao, 6, rng)
    dms = np.stack([L @ R.T for L, R in pairs])
    K_ref = _dense_gpu_k(cell, dms)
    K1 = get_k_lowrank(cell, cell.mesh, pairs, row_blk=3, rank_blk=2,
                       group=DeviceGroup([0]), verbose=False)
    K2 = get_k_lowrank(cell, cell.mesh, pairs, row_blk=3, rank_blk=2,
                       group=DeviceGroup(_two_slots()), verbose=False)
    tel = get_last_telemetry()
    scale = np.maximum(np.abs(K_ref).max(axis=(1, 2)), 1.0)
    assert np.all(np.abs(K1 - K_ref).max(axis=(1, 2)) < 1e-9 * scale)
    assert np.all(np.abs(K2 - K_ref).max(axis=(1, 2)) < 1e-9 * scale)
    assert np.abs(K1 - K2).max() < 1e-12 * scale.max()
    assert tel["gpu_slots"] == _two_slots()
    assert all(sum(s["multi_gpu"]["tasks_per_slot"]) == len(range(0, cell.nao, 3))
               for s in tel["sets"])


def test_ket_projected_matches_dense_with_and_without_ao_grid():
    """K @ ket from r x nk transforms (no AO grid resident) == dense K @ ket."""
    _need_cupy()
    import cupy as cp
    from lib_pprpa.gpu_fft_k import get_k_lowrank, get_last_telemetry, ao_on_grid
    from lib_pprpa.gpu_multi import DeviceGroup
    cell = _diamond_cell()
    rng = np.random.default_rng(5)
    nao = cell.nao
    pairs = _factors(nao, 6, rng)
    ket = rng.standard_normal((nao, 4))
    dms = np.stack([L @ R.T for L, R in pairs])
    K_ref = _dense_gpu_k(cell, dms) @ ket                       # (3, nao, 4)
    scale = np.abs(K_ref).max(axis=(1, 2))
    # grid-chunked AO contraction (production: the AO grid is never held)
    Kk = get_k_lowrank(cell, cell.mesh, pairs, ket=ket, verbose=False, group=DeviceGroup([0]))
    assert Kk.shape == (3, nao, 4)
    assert np.all(np.abs(Kk - K_ref).max(axis=(1, 2)) < 1e-9 * np.maximum(scale, 1.0))
    tel = get_last_telemetry()
    assert tel["ket"] and tel["sets"][0]["transforms"] == 4 * 6
    # with a resident AO grid, forced tiny blocks, two virtual slots
    ao = ao_on_grid(cell, cell.mesh)
    Kk2 = get_k_lowrank(cell, cell.mesh, pairs, ket=ket, ao=ao, row_blk=1, fft_blk=4,
                        verbose=False, group=DeviceGroup(_two_slots()))
    assert np.all(np.abs(Kk2 - K_ref).max(axis=(1, 2)) < 1e-9 * np.maximum(scale, 1.0))
    assert np.abs(Kk2 - Kk).max() < 1e-11 * max(scale.max(), 1.0)
    # C2C chain agrees with the R2C default
    Kc = get_k_lowrank(cell, cell.mesh, pairs, ket=ket, rfft=False, verbose=False,
                       group=DeviceGroup([0]))
    assert np.abs(Kc - Kk).max() < 1e-11 * max(scale.max(), 1.0)
    # single pair, ket
    K1 = get_k_lowrank(cell, cell.mesh, pairs[2], ket=ket, verbose=False, group=DeviceGroup([0]))
    assert K1.shape == (nao, 4) and np.abs(K1 - K_ref[2]).max() < 1e-9 * max(scale[2], 1.0)


if __name__ == "__main__":
    _need_cupy()
    test_ket_projected_matches_dense_with_and_without_ao_grid()
    print("OK  test_ket_projected_matches_dense_with_and_without_ao_grid")
    for name, fn in [("auto blocks", test_lowrank_matches_dense_auto_blocks),
                     ("forced small blocks", test_lowrank_matches_dense_forced_small_blocks),
                     ("supercell high rank", test_lowrank_rank_exceeds_nothing_supercell),
                     ("virtual two slots", test_virtual_two_slots_match_one_slot)]:
        fn()
        print(f"OK  {name}")
    cell = _diamond_cell()
    err, scale, err_one, tel = _run_case(cell, rank=5)
    print(f"nao={cell.nao} mesh={list(cell.mesh)} max|K_lr-K_dense| per set = {err}  "
          f"(|K| = {scale});  single-pair err = {err_one:.3e};  telemetry = {tel}")
