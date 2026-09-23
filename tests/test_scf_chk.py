"""SCF checkpoint reuse (lib_pprpa.scf_chk): roundtrip, path resolution, and
the mismatches that must be refused rather than silently reused.

CPU only -- no GPU, no gpu4pyscf.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from lib_pprpa import scf_chk

ENVS = (scf_chk.ENV_BOTH, scf_chk.ENV_IN, scf_chk.ENV_OUT)


def _clear_env():
    for e in ENVS:
        os.environ.pop(e, None)


def _cell(dz=0.0, basis="sto-3g", a=3.0, charge=0, symbols=("H", "H")):
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = [[symbols[0], [0.0, 0.0, 0.0]], [symbols[1], [0.0, 0.0, 0.83 + dz]]]
    cell.a = np.eye(3) * a
    cell.basis = basis
    cell.charge = charge
    cell.ke_cutoff = 40.0
    cell.verbose = 0
    cell.build()
    return cell


def _dm(cell, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((cell.nao, cell.nao))
    return A + A.T


def test_roundtrip_and_displacement(tmp_path):
    _clear_env()
    ref, moved = _cell(), _cell(dz=0.02)
    dm = _dm(ref)
    p = str(tmp_path / "scf.npz")
    assert scf_chk.save_dm(ref, dm, path=p, e_tot=-1.25, verbose=False) == p
    back = scf_chk.load_dm0(moved, path=p, verbose=False)
    assert back is not None
    assert np.array_equal(back, dm)
    _, coords, _, meta = scf_chk.read(p)
    assert meta["e_tot"] == -1.25 and meta["format"] == scf_chk.FORMAT
    assert np.allclose(coords, ref.atom_coords())
    # the temporary file is gone: the write is atomic
    assert [f for f in os.listdir(tmp_path) if ".tmp" in f] == []


def test_missing_and_corrupt_files_fall_back(tmp_path):
    _clear_env()
    cell = _cell()
    assert scf_chk.load_dm0(cell, path=str(tmp_path / "nope.npz"), verbose=False) is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    assert scf_chk.load_dm0(cell, path=str(bad), verbose=False) is None


def test_mismatched_cells_are_refused(tmp_path):
    _clear_env()
    ref = _cell()
    p = str(tmp_path / "scf.npz")
    scf_chk.save_dm(ref, _dm(ref), path=p, verbose=False)
    # charges/species are chosen to keep nelectron even, so the cells build
    for other in (_cell(basis="6-31g"),          # different nao / basis
                  _cell(a=4.0),                  # different lattice
                  _cell(charge=-2),              # different electron count
                  _cell(symbols=("He", "He"))):  # different species, same nao
        assert scf_chk.load_dm0(other, path=p, verbose=False) is None, other.atom
    # same cell: accepted
    assert scf_chk.load_dm0(_cell(), path=p, verbose=False) is not None


def test_max_displacement_guard(tmp_path):
    _clear_env()
    ref = _cell()
    p = str(tmp_path / "scf.npz")
    scf_chk.save_dm(ref, _dm(ref), path=p, verbose=False)
    far = _cell(dz=1.5)
    assert scf_chk.load_dm0(far, path=p, max_disp_bohr=0.1, verbose=False) is None
    assert scf_chk.load_dm0(far, path=p, max_disp_bohr=None, verbose=False) is not None


def test_path_resolution():
    _clear_env()
    try:
        assert scf_chk.resolve_paths() == (None, None)
        os.environ[scf_chk.ENV_BOTH] = "roll.npz"
        assert scf_chk.resolve_paths() == ("roll.npz", "roll.npz")   # rolling: opt
        os.environ[scf_chk.ENV_IN] = "ref.npz"
        assert scf_chk.resolve_paths() == ("ref.npz", None)          # read-only: displacements
        os.environ[scf_chk.ENV_OUT] = "out.npz"
        assert scf_chk.resolve_paths() == ("ref.npz", "out.npz")
        # explicit arguments win; False disables a direction
        assert scf_chk.resolve_paths(load="a", save="b") == ("a", "b")
        assert scf_chk.resolve_paths(load=False, save=False) == (None, None)
    finally:
        _clear_env()


def test_save_dm_without_a_path_is_a_noop(tmp_path):
    _clear_env()
    cell = _cell()
    assert scf_chk.save_dm(cell, _dm(cell), verbose=False) is None


def test_run_scf_reuses_and_writes(tmp_path):
    """End to end on a real (tiny, CPU) Gamma-point SCF: the second run starts
    from the first one's density and lands on the same energy."""
    _clear_env()
    from pyscf.pbc import dft as cdft
    p = str(tmp_path / "scf.npz")
    cell = _cell()

    mf1 = cdft.RKS(cell, xc="lda,vwn")
    mf1.exxdiv = None
    mf1.conv_tol = 1e-9
    scf_chk.run_scf(mf1, cell, save=p, load=False, verbose=False)
    assert mf1.converged and os.path.exists(p)

    mf2 = cdft.RKS(cell, xc="lda,vwn")
    mf2.exxdiv = None
    mf2.conv_tol = 1e-9
    scf_chk.run_scf(mf2, cell, load=p, save=False, verbose=False)
    assert mf2.converged
    assert abs(float(mf2.e_tot) - float(mf1.e_tot)) < 1e-8
    # starting from the converged density: fewer cycles than from scratch
    assert mf2.scf_summary  # sanity: the kernel really ran


def test_unconverged_scf_is_not_checkpointed(tmp_path):
    _clear_env()
    from pyscf.pbc import dft as cdft
    p = str(tmp_path / "scf.npz")
    cell = _cell()
    mf = cdft.RKS(cell, xc="lda,vwn")
    mf.exxdiv = None
    mf.max_cycle = 1
    mf.conv_tol = 1e-12
    scf_chk.run_scf(mf, cell, save=p, load=False, verbose=False)
    assert not mf.converged
    assert not os.path.exists(p)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for fn in (test_roundtrip_and_displacement, test_missing_and_corrupt_files_fall_back,
                   test_mismatched_cells_are_refused, test_max_displacement_guard,
                   test_save_dm_without_a_path_is_a_noop, test_run_scf_reuses_and_writes,
                   test_unconverged_scf_is_not_checkpointed):
            sub = d / fn.__name__
            sub.mkdir()
            fn(sub)
            print("OK ", fn.__name__)
        test_path_resolution()
        print("OK  test_path_resolution")
