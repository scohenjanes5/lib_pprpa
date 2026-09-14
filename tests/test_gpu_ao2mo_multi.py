"""gpu_ao2mo strips on one slot vs a virtual two-slot DeviceGroup vs CPU pyscf.

Small diamond cell (gth-szv, mesh 20^3) with pair_blk forced small so there
are many strips.  Both groups must agree with pyscf ``get_mo_eri`` and with
each other to round-off.
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


def _setup():
    from pyscf.pbc import gto, dft as cdft
    a0 = 3.370137329
    cell = gto.M(
        atom=[["C", [0.0, 0.0, 0.0]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
        a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
        unit="bohr", basis="gth-szv", pseudo="gth-pade", verbose=0)
    cell.mesh = [20, 20, 20]
    cell.build()
    mf = cdft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    nocc = cell.nelectron // 2
    nmo = cell.nao
    cocc = mf.mo_coeff[:, :nocc]
    cvir = mf.mo_coeff[:, nocc:]
    eri = mf.with_df.get_mo_eri(mf.mo_coeff, compact=False)
    eri = eri.reshape(nmo, nmo, nmo, nmo).transpose(0, 2, 1, 3)
    ref = {"vvvv": eri[nocc:, nocc:, nocc:, nocc:],
           "oovv": eri[:nocc, :nocc, nocc:, nocc:],
           "oooo": eri[:nocc, :nocc, :nocc, :nocc]}
    return cell, cocc, cvir, ref


def _run(cell, cocc, cvir, group, pair_blk):
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks, get_last_telemetry
    vvvv, oovv, oooo = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, pair_blk=pair_blk,
                                        group=group)
    return {"vvvv": np.asarray(vvvv), "oovv": np.asarray(oovv), "oooo": np.asarray(oooo)}, \
        get_last_telemetry()


def test_one_slot_vs_two_slots_vs_cpu():
    _need_cupy()
    from lib_pprpa.gpu_multi import DeviceGroup
    cell, cocc, cvir, ref = _setup()
    nvir = cvir.shape[1]
    got1, tel1 = _run(cell, cocc, cvir, DeviceGroup([0]), pair_blk=nvir)   # one MO row per strip
    got2, tel2 = _run(cell, cocc, cvir, DeviceGroup(_two_slots()), pair_blk=nvir)
    for name in ("vvvv", "oovv", "oooo"):
        assert np.abs(got1[name] - ref[name]).max() < 1e-11, name
        assert np.abs(got2[name] - ref[name]).max() < 1e-11, name
        assert np.abs(got1[name] - got2[name]).max() < 1e-13, name
    assert tel1["gpu_slots"] == [0] and tel2["gpu_slots"] == _two_slots()
    blocks2 = {b["name"]: b for b in tel2["blocks"]}
    assert all(b["output_location"] != "gpu" for b in blocks2.values())   # >1 slot: host-staged
    assert sum(blocks2["vvvv"]["multi_gpu"]["tasks_per_slot"]) == blocks2["vvvv"]["nstrips"]


def test_auto_pair_blk_and_return_gpu():
    _need_cupy()
    import cupy as cp
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks
    cell, cocc, cvir, ref = _setup()
    out = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh, return_gpu=True, group=DeviceGroup([0]))
    for name, arr in zip(("vvvv", "oovv", "oooo"), out):
        assert np.abs(cp.asnumpy(arr) - ref[name]).max() < 1e-11, name


if __name__ == "__main__":
    _need_cupy()
    test_one_slot_vs_two_slots_vs_cpu()
    print("OK  test_one_slot_vs_two_slots_vs_cpu")
    test_auto_pair_blk_and_return_gpu()
    print("OK  test_auto_pair_blk_and_return_gpu")
