"""Low-rank FFT pairing-K force (lib_pprpa.gpu_pairing_force) vs the CPU reference.

Reference is exactly the pairing term of lib_pprpa/grad/pprpa_gamma.py:
    vk = kmf_grad.get_k(np.array([X]))[:, 0]
    de[A] += 2 * einsum('xij,ij->x', vk[:, p0:p1], X[p0:p1, :])
for X = L @ R.T with symmetric, antisymmetric and general low-rank cores, and
for forced tiny strips (crossing pair-row boundaries), the all-pairs variant,
and a virtual two-slot device group on one GPU.
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


def _diamond_cell(basis="gth-szv", ke=100.0, nrep=1):
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


def _cpu_reference(cell, X):
    """CPU pyscf FFT reference for the pairing term, per atom (natm, 3)."""
    from pyscf.pbc import dft
    kmf = dft.KRKS(cell, kpts=np.zeros((1, 3)))
    kmf.xc = "pbe"
    kmf.exxdiv = None
    kmf_grad = kmf.nuc_grad_method()
    vk = kmf_grad.get_k(np.array([X]))[:, 0]           # (3, nao, nao)
    aoslices = cell.aoslice_by_atom()
    de = np.zeros((cell.natm, 3))
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia, 2:]
        de[ia] += np.einsum('xij,ij->x', vk[:, p0:p1], X[p0:p1, :]) * 2
    return de


def _factors(nao, rank, rng):
    C = rng.standard_normal((nao, rank))
    x = rng.standard_normal((rank, rank))
    return {
        "sym": (C @ ((x + x.T) * 0.5), C),
        "anti": (C @ ((x - x.T) * 0.5), C),
        "gen": (rng.standard_normal((nao, rank)), rng.standard_normal((nao, rank))),
    }


def _check(cell, L, R, de_ref, **kw):
    from lib_pprpa.gpu_pairing_force import pairing_k_force_lowrank
    de = pairing_k_force_lowrank(cell, cell.mesh, L, R, verbose=False, **kw)
    scale = max(np.abs(de_ref).max(), 1e-8)
    err = np.abs(de - de_ref).max() / scale
    return de, err


def test_matches_cpu_reference_all_cores():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    cell = _diamond_cell()
    rng = np.random.default_rng(0)
    for name, (L, R) in _factors(cell.nao, 5, rng).items():
        X = L @ R.T
        de_ref = _cpu_reference(cell, X)
        assert np.abs(de_ref).max() > 1e-6, "reference term unexpectedly tiny"
        de, err = _check(cell, L, R, de_ref, group=DeviceGroup([0]))
        assert err < 1e-9, (name, err, de, de_ref)


def test_forced_small_strips_and_all_pairs_variant():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    cell = _diamond_cell()
    rng = np.random.default_rng(1)
    L, R = _factors(cell.nao, 7, rng)["gen"]
    de_ref = _cpu_reference(cell, L @ R.T)
    g = DeviceGroup([0])
    de_auto, err_auto = _check(cell, L, R, de_ref, group=g)
    de_tiny, err_tiny = _check(cell, L, R, de_ref, group=g, blk=3)       # crosses rows
    de_all, err_all = _check(cell, L, R, de_ref, group=g, symmetric=False, blk=5)
    for err in (err_auto, err_tiny, err_all):
        assert err < 1e-9, (err_auto, err_tiny, err_all)
    assert np.abs(de_tiny - de_auto).max() < 1e-12 * max(np.abs(de_auto).max(), 1.0)
    assert np.abs(de_all - de_auto).max() < 1e-11 * max(np.abs(de_auto).max(), 1.0)


def test_virtual_two_slots_match_single_slot():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.gpu_pairing_force import get_last_telemetry
    cell = _diamond_cell(nrep=2)
    rng = np.random.default_rng(2)
    L, R = _factors(cell.nao, cell.nao - 3, rng)["anti"]    # rank close to nao
    de_ref = _cpu_reference(cell, L @ R.T)
    de1, err1 = _check(cell, L, R, de_ref, group=DeviceGroup([0]), blk=4)
    de2, err2 = _check(cell, L, R, de_ref, group=DeviceGroup(_two_slots()), blk=4)
    tel = get_last_telemetry()
    assert err1 < 1e-9 and err2 < 1e-9, (err1, err2)
    assert np.abs(de1 - de2).max() < 1e-11 * max(np.abs(de1).max(), 1.0)
    assert tel["multi_gpu"]["nslots"] == 2 and sum(tel["multi_gpu"]["tasks_per_slot"]) == tel["nstrips"]


def test_zero_columns_dropped_and_rank_zero():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.gpu_pairing_force import pairing_k_force_lowrank, get_last_telemetry
    cell = _diamond_cell()
    rng = np.random.default_rng(3)
    L, R = _factors(cell.nao, 4, rng)["gen"]
    Lz = np.hstack([L, np.zeros((cell.nao, 2))])
    Rz = np.hstack([R, rng.standard_normal((cell.nao, 2))])
    de_ref = _cpu_reference(cell, Lz @ Rz.T)
    de = pairing_k_force_lowrank(cell, cell.mesh, Lz, Rz, group=DeviceGroup([0]), verbose=False)
    assert get_last_telemetry()["rank"] == 4
    assert np.abs(de - de_ref).max() < 1e-9 * max(np.abs(de_ref).max(), 1.0)
    de0 = pairing_k_force_lowrank(cell, cell.mesh, np.zeros((cell.nao, 3)), Rz[:, :3],
                                  group=DeviceGroup([0]), verbose=False)
    assert de0.shape == (cell.natm, 3) and np.all(de0 == 0)


if __name__ == "__main__":
    _need_cupy()
    for fn in (test_matches_cpu_reference_all_cores,
               test_forced_small_strips_and_all_pairs_variant,
               test_virtual_two_slots_match_single_slot,
               test_zero_columns_dropped_and_rank_zero):
        fn()
        print(f"OK  {fn.__name__}")
