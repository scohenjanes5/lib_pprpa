"""Tiled, multi-GPU ``gpu_ao2mo._mos_on_grid`` vs the untiled AO contraction.

The MO grid is the one place ``nao`` enters ao2mo, and the untiled form holds an
(ngrid, nao) AO array (90 GB at nao=2795 on 159^3).  ``_mos_on_grid`` evaluates
the AOs in grid chunks and leaves one (nmo, ngrid) array per coefficient set in
every slot's ``DeviceGroup`` state.  Chunking is exact -- an output element sums
over ``nao`` only, so no reduction crosses a chunk boundary -- hence these tests
demand agreement at round-off, not a tolerance.

Both multi-slot strategies are covered: the default ``replicated`` build (every
slot chunks the whole grid itself) and the opt-in ``dispatched`` one (chunks
split across slots through a host array).

Needs a GPU; two virtual slots on one device cover the multi-device path.
"""
from __future__ import annotations

import os

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


_CACHE = {}


def _setup():
    """Small diamond cell with a basis big enough that nao > nmo per set."""
    if "cell" in _CACHE:
        return _CACHE["cell"], _CACHE["cocc"], _CACHE["cvir"], _CACHE["ref"]
    from pyscf.pbc import gto, dft as cdft
    a0 = 3.370137329
    cell = gto.M(
        atom=[["C", [0.0, 0.0, 0.0]], ["C", [a0 / 2, a0 / 2, a0 / 2]]],
        a=np.array([[0, a0, a0], [a0, 0, a0], [a0, a0, 0]]),
        unit="bohr", basis="gth-dzvp", pseudo="gth-pade", verbose=0)
    cell.mesh = [21, 21, 21]
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
    _CACHE.update(cell=cell, cocc=cocc, cvir=cvir, ref=ref)
    return cell, cocc, cvir, ref


def _reference_grids(cell, mos):
    """Untiled (ngrid, nao) AO array -- exactly what _mo_on_grid used to build."""
    import cupy as cp
    from gpu4pyscf.pbc.dft import numint as gnumint
    coords = cell.gen_uniform_grids(cell.mesh)
    ao = gnumint.eval_ao_kpts(cell, coords, kpts=np.zeros((1, 3)), deriv=0)[0]
    ao = cp.asarray(ao)
    return [cp.asnumpy(cp.asarray(m).T @ ao.T) for m in mos]


def _grids_from(group, keys):
    """[(rank, key) -> numpy] for every slot, so we can check each slot's copy."""
    import cupy as cp
    out = []
    for ctx in group.ctxs:
        with cp.cuda.Device(ctx.device_id):
            out.append([cp.asnumpy(ctx.state[k]) for k in keys])
    return out


def test_tiled_mo_grid_matches_untiled():
    """Default chunk, forced chunks, and the env override all reproduce the untiled grid."""
    _need_cupy()
    import cupy as cp
    from lib_pprpa.gpu_ao2mo import _mos_on_grid, _mo_on_grid
    from lib_pprpa.gpu_multi import DeviceGroup

    cell, cocc, cvir, _ref = _setup()
    ref_o, ref_v = _reference_grids(cell, [cocc, cvir])
    ngrid = int(np.prod(cell.mesh))
    one = DeviceGroup([0], label="one")

    for gchunk in (None, 512, 1):
        stats = _mos_on_grid(cell, [cocc, cvir], ["o", "v"], cell.mesh,
                             group=one, gchunk=gchunk)
        assert stats["mode"] == "replicated"
        (o, v), = _grids_from(one, ["o", "v"])
        assert np.abs(o - ref_o).max() < 1e-12
        assert np.abs(v - ref_v).max() < 1e-12
        if gchunk is not None:
            assert stats["gchunk"] == min(gchunk, ngrid)
            assert stats["nchunks"][0] == -(-ngrid // min(gchunk, ngrid))
        one.free(["o", "v"])

    os.environ["GPU_AO2MO_GRID_CHUNK"] = "97"
    try:
        stats = _mos_on_grid(cell, [cocc, cvir], ["o", "v"], cell.mesh, group=one)
        assert stats["gchunk"] == 97
        (o, _v), = _grids_from(one, ["o", "v"])
        assert np.abs(o - ref_o).max() < 1e-12
        one.free(["o", "v"])
    finally:
        del os.environ["GPU_AO2MO_GRID_CHUNK"]

    got = _mo_on_grid(cell, cocc, cell.mesh, group=one)
    assert np.abs(cp.asnumpy(got) - ref_o).max() < 1e-12


def test_replicated_build_gives_every_slot_the_full_grid():
    """Default multi-slot path: no host staging, and each slot holds the whole grid."""
    _need_cupy()
    from lib_pprpa.gpu_ao2mo import _mos_on_grid
    from lib_pprpa.gpu_multi import DeviceGroup

    cell, cocc, cvir, _ref = _setup()
    ref_o, ref_v = _reference_grids(cell, [cocc, cvir])
    two = DeviceGroup(_two_slots(), label="two")

    stats = _mos_on_grid(cell, [cocc, cvir], ["o", "v"], cell.mesh,
                         group=two, gchunk=256)
    assert stats["mode"] == "replicated"
    assert len(stats["nchunks"]) == two.nslots
    assert all(n == stats["nchunks"][0] for n in stats["nchunks"]), \
        "every slot builds the whole grid"

    per_slot = _grids_from(two, ["o", "v"])
    assert len(per_slot) == two.nslots
    for rank, (o, v) in enumerate(per_slot):
        assert np.abs(o - ref_o).max() < 1e-12, f"slot {rank} occ"
        assert np.abs(v - ref_v).max() < 1e-12, f"slot {rank} vir"
    two.free(["o", "v"])


def test_dispatched_build_matches_replicated():
    """Opt-in chunk dispatch: slots share the work and still agree exactly."""
    _need_cupy()
    from lib_pprpa.gpu_ao2mo import _mos_on_grid
    from lib_pprpa.gpu_multi import DeviceGroup

    cell, cocc, cvir, _ref = _setup()
    ref_o, ref_v = _reference_grids(cell, [cocc, cvir])
    two = DeviceGroup(_two_slots(), label="two")

    stats = _mos_on_grid(cell, [cocc, cvir], ["o", "v"], cell.mesh,
                         group=two, gchunk=256, dispatch=True)
    assert stats["mode"] == "dispatched"
    tps = stats["multi_gpu"]["tasks_per_slot"]
    assert sum(tps) == stats["nchunks"][0], "task boundaries independent of slots"
    assert all(n > 0 for n in tps), "both slots must take work"

    for rank, (o, v) in enumerate(_grids_from(two, ["o", "v"])):
        assert np.abs(o - ref_o).max() < 1e-12, f"slot {rank} occ"
        assert np.abs(v - ref_v).max() < 1e-12, f"slot {rank} vir"
    two.free(["o", "v"])

    # the env switch selects the same path
    os.environ["GPU_AO2MO_GRID_DISPATCH"] = "1"
    try:
        stats = _mos_on_grid(cell, [cocc, cvir], ["o", "v"], cell.mesh,
                             group=two, gchunk=256)
        assert stats["mode"] == "dispatched"
    finally:
        del os.environ["GPU_AO2MO_GRID_DISPATCH"]
    two.free(["o", "v"])


def test_ao2mo_blocks_with_tiny_grid_chunk():
    """The full transform still reproduces pyscf when the AO grid is chunked hard."""
    _need_cupy()
    from lib_pprpa.gpu_ao2mo import gpu_ao2mo_blocks, get_last_telemetry
    from lib_pprpa.gpu_multi import DeviceGroup

    cell, cocc, cvir, ref = _setup()

    os.environ["GPU_AO2MO_GRID_CHUNK"] = "333"
    try:
        for devices in ([0], _two_slots()):
            for dispatch in ("0", "1"):
                if devices == [0] and dispatch == "1":
                    continue  # dispatch is a no-op on one slot
                os.environ["GPU_AO2MO_GRID_DISPATCH"] = dispatch
                group = DeviceGroup(devices, label=f"g{len(devices)}")
                vvvv, oovv, oooo = gpu_ao2mo_blocks(cell, cocc, cvir, cell.mesh,
                                                    pair_blk=4, group=group)
                tel = get_last_telemetry()
                assert tel["mo_grid"]["gchunk"] == 333
                assert (tel["mo_grid"]["ao_block_bytes"]
                        < tel["mo_grid"]["ao_untiled_bytes"])
                for name, got in (("vvvv", vvvv), ("oovv", oovv), ("oooo", oooo)):
                    err = np.abs(np.asarray(got) - np.asarray(ref[name])).max()
                    assert err < 1e-9, (f"{name} off by {err:.3e} on "
                                        f"{len(devices)} slot(s), dispatch={dispatch}")
    finally:
        os.environ.pop("GPU_AO2MO_GRID_CHUNK", None)
        os.environ.pop("GPU_AO2MO_GRID_DISPATCH", None)


if __name__ == "__main__":
    test_tiled_mo_grid_matches_untiled()
    print("tiled vs untiled: ok")
    test_replicated_build_gives_every_slot_the_full_grid()
    print("replicated two slots: ok")
    test_dispatched_build_matches_replicated()
    print("dispatched two slots: ok")
    test_ao2mo_blocks_with_tiny_grid_chunk()
    print("ao2mo blocks with chunked grid: ok")
