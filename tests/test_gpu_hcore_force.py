"""Density-space hcore force (grad.gpu_hcore_force) vs the per-atom
``krhf.hcore_generator`` loop it replaces (pprpa_gamma_gpu._hcore_force).

Both branches are covered: pseudo-potential (gth-pade, the production case,
``eval_vpplocG_SI_gradient``) and all-electron (``eval_nucG_SI_gradient``).
The all-electron cell is physically meaningless at this cutoff -- the test
compares two evaluations of the same formula, not the formula against nature.
"""
from __future__ import annotations

import numpy as np

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

GAMMA = np.zeros((1, 3))


def _need_cupy():
    try:
        import cupy as _cp
        _cp.cuda.runtime.getDevice()
    except Exception:
        if pytest is None:
            raise RuntimeError("CUDA GPU required for this test") from None
        pytest.skip("CUDA GPU required")


def _pseudo_cell_and_mf():
    """Diamond with both atoms off the symmetric positions, so every atom has a
    non-zero force and a wrong sign or a transposed slice cannot hide."""
    from pyscf.pbc import gto, dft as cdft
    a0 = 3.5668
    cell = gto.Cell()
    cell.atom = [["C", [0.02, -0.01, 0.03]],
                 ["C", [a0 / 4 + 0.05, a0 / 4 - 0.02, a0 / 4 + 0.01]]]
    cell.a = np.array([[0, a0 / 2, a0 / 2], [a0 / 2, 0, a0 / 2], [a0 / 2, a0 / 2, 0]])
    cell.basis = "gth-dzvp"
    cell.pseudo = "gth-pade"
    cell.ke_cutoff = 120.0
    cell.verbose = 0
    cell.build()
    mf = cdft.RKS(cell, xc="pbe")
    mf.exxdiv = None
    mf.conv_tol = 1e-9
    mf.kernel()
    return cell, mf


def _allelectron_cell_and_mf():
    from pyscf.pbc import gto, dft as cdft
    cell = gto.Cell()
    cell.atom = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.83]]]
    cell.a = np.eye(3) * 3.0
    cell.basis = "sto-3g"
    cell.ke_cutoff = 60.0
    cell.verbose = 0
    cell.build()
    mf = cdft.RKS(cell, xc="lda,vwn")
    mf.exxdiv = None
    mf.conv_tol = 1e-9
    mf.kernel()
    return cell, mf


def _density(mf, cell, seed, symmetric=True):
    """D plus a perturbation: the production T = D + P is symmetric but is not
    the SCF density.  ``symmetric=False`` drops the symmetry, which the
    ``hermi=0`` contraction of the AO-derivative term has to survive."""
    D = np.asarray(mf.make_rdm1(), dtype=np.float64)
    if D.ndim == 3:
        D = D[0]
    A = np.random.default_rng(seed).standard_normal((cell.nao, cell.nao))
    return D + ((A + A.T) if symmetric else A) * 0.01


def _check(cell, mf, seed=1, symmetric=True, slots=None, tol=1e-9):
    from lib_pprpa.gpu_multi import DeviceGroup
    from lib_pprpa.grad.gpu_hcore_force import hcore_force
    from lib_pprpa.grad.pprpa_gamma_gpu import _hcore_force, _make_kmf
    T = _density(mf, cell, seed, symmetric)
    gg = _make_kmf(cell, GAMMA, mf, True).nuc_grad_method()
    ref, _ = _hcore_force(gg, cell, GAMMA, mf, True, T, DeviceGroup([0]))
    group = DeviceGroup(slots) if slots else None
    new, stats = hcore_force(cell, GAMMA, T, group=group, verbose=False)
    assert new.shape == ref.shape == (cell.natm, 3)
    scale = max(np.abs(ref).max(), 1e-12)
    err = np.abs(new - ref).max() / scale
    assert err < tol, (seed, symmetric, err, ref, new)
    assert stats["natm"] == cell.natm
    return err


def test_pseudo_matches_generator():
    _need_cupy()
    err = _check(*_pseudo_cell_and_mf())
    print(f"pseudo: rel err {err:.2e}", flush=True)


def test_all_electron_matches_generator():
    _need_cupy()
    err = _check(*_allelectron_cell_and_mf(), seed=7)
    print(f"all-electron: rel err {err:.2e}", flush=True)


def test_non_symmetric_density_matches_generator():
    """Only the symmetric part of T reaches rho_T, but the AO-derivative term
    sees all of it -- hence hermi=0 in the contract_h1e_dm call."""
    _need_cupy()
    err = _check(*_pseudo_cell_and_mf(), seed=5, symmetric=False)
    print(f"non-symmetric T: rel err {err:.2e}", flush=True)


def test_device_group_is_accepted_and_inert():
    """The kernels are gpu4pyscf's own and single-device; passing the force
    assembly's group must be harmless, not an error or a different answer."""
    _need_cupy()
    import cupy as cp
    cell, mf = _pseudo_cell_and_mf()
    slots = [0, 1] if cp.cuda.runtime.getDeviceCount() >= 2 else [0, 0]
    assert _check(cell, mf, seed=2) == _check(cell, mf, seed=2, slots=slots)


def _e2e_module():
    """The end-to-end gradient test's helpers, imported by path so this works
    whatever pytest's import mode is."""
    import importlib
    import os
    import sys
    d = os.path.dirname(os.path.abspath(__file__))
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module("test_gpu_grad_pairing_e2e")


def test_end_to_end_gradient_matches_the_dense_path():
    """The whole force through grad_elec, default vs PPRPA_HCORE=dense: checks
    that the density-space path is what actually runs and that swapping it out
    changes nothing but the wall time."""
    _need_cupy()
    import os
    from lib_pprpa.gpu_multi import DeviceGroup
    e2e = _e2e_module()
    cell = e2e._build_cell()
    kg, mf, mp, xy = e2e._scf_and_pprpa(cell, "t")
    de_new = e2e._gpu_grad(mp, mf, cell, "t", xy, "lowrank", DeviceGroup([0]))
    os.environ["PPRPA_HCORE"] = "dense"
    try:
        de_ref = e2e._gpu_grad(mp, mf, cell, "t", xy, "lowrank", DeviceGroup([0]))
    finally:
        os.environ.pop("PPRPA_HCORE", None)
    scale = max(np.abs(de_ref).max(), 1e-3)
    err = np.abs(de_new - de_ref).max() / scale
    print(f"e2e gradient: |de|max={scale:.3e}  density-vs-dense={err:.2e}", flush=True)
    assert err < 1e-9, err


def test_env_switch_selects_the_reference_path():
    import os
    from lib_pprpa.grad.gpu_hcore_force import use_density_path
    assert use_density_path()
    os.environ["PPRPA_HCORE"] = "dense"
    try:
        assert not use_density_path()
    finally:
        os.environ.pop("PPRPA_HCORE", None)
    assert use_density_path()


if __name__ == "__main__":
    _need_cupy()
    test_env_switch_selects_the_reference_path()
    print("OK  test_env_switch_selects_the_reference_path")
    test_pseudo_matches_generator()
    test_all_electron_matches_generator()
    test_non_symmetric_density_matches_generator()
    test_device_group_is_accepted_and_inert()
    print("OK  device group inert")
    test_end_to_end_gradient_matches_the_dense_path()
