"""Reuse a converged SCF density as the next SCF's initial guess.

Why
---
The force workflows run the same cell many times at nearby geometries: every
step of a geometry optimization, and -- far more of them -- the 200+ phonopy
displacements, each one atom away from the *same* optimized geometry.  Nothing
about the gradient itself can be carried across those runs (the relaxed density
changes everywhere when any atom moves), but the SCF's starting point can: an
AO density matrix from a geometry 0.01 A away is a much better guess than the
atomic superposition, and it costs one file to pass along.

This only changes the path the SCF takes, never where it lands --
``run_scf`` still refuses to hand back an unconverged object.  The one thing a
guess *can* change is *which* self-consistent solution is reached when a system
has more than one (a charged defect can), and there reuse is the conservative
choice: every displacement starts from the reference state instead of
rediscovering one from scratch.

Use
---
``run_scf(kg, cell)`` in place of ``kg.kernel()``, with the path from an
argument or the environment:

* ``PPRPA_SCF_CHK``      -- read and write (a rolling checkpoint; optimization)
* ``PPRPA_SCF_CHK_IN``   -- read only, overrides the above (displacements: all
  of them read the optimized geometry's checkpoint, none of them write)
* ``PPRPA_SCF_CHK_OUT``  -- write only, overrides the above

A checkpoint carries the geometry it came from, and is refused -- with a line
saying why, never an exception -- when the cell it is handed does not match in
the ways an AO density matrix depends on: same atoms in the same order, same
lattice, same basis, pseudo, charge and spin.  A different geometry is the
point, so only the displacement is reported, not rejected (unless
``max_disp_bohr`` is given).  ``ke_cutoff`` and the mesh may differ freely: an
AO density matrix does not know about the grid.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

FORMAT = "lib_pprpa.scf_chk/1"

ENV_BOTH = "PPRPA_SCF_CHK"
ENV_IN = "PPRPA_SCF_CHK_IN"
ENV_OUT = "PPRPA_SCF_CHK_OUT"

BOHR = 0.52917721092


def _log(msg):
    print(f"[scf_chk] {msg}", flush=True)


def _env(name):
    v = os.environ.get(name, "")
    v = v.strip()
    return v or None


def resolve_paths(load=None, save=None):
    """``(load_path, save_path)`` from explicit arguments, else the environment.

    An explicit argument always wins; ``False`` disables that direction.
    """
    both = _env(ENV_BOTH)
    if load is None:
        load = _env(ENV_IN) or both
    if save is None:
        save = _env(ENV_OUT) or (None if _env(ENV_IN) else both)
    return (load or None) if load is not False else None, \
           (save or None) if save is not False else None


def signature(cell):
    """The properties an AO density matrix is tied to (geometry excluded)."""
    return {
        "symbols": [cell.atom_symbol(i) for i in range(cell.natm)],
        "nao": int(cell.nao),
        "nelectron": int(cell.nelectron),
        "charge": int(cell.charge),
        "spin": int(cell.spin),
        "basis": repr(cell.basis),
        "pseudo": repr(cell.pseudo),
        "ecp": repr(getattr(cell, "ecp", None)),
    }


def _mismatch(have, want):
    """First signature key that differs, as a human-readable string, else None."""
    for k in want:
        if have.get(k) != want[k]:
            return f"{k}: checkpoint {have.get(k)!r} != cell {want[k]!r}"
    return None


def _to_numpy(a):
    """numpy view of a numpy or cupy array."""
    return np.asarray(a.get() if hasattr(a, "get") else a)


def save_dm(cell, dm, path=None, e_tot=None, xc=None, verbose=True):
    """Write ``dm`` (any shape ending in ``(nao, nao)``) plus the geometry it was
    converged at.  Returns the path, or None when no save path is configured.

    The write is atomic (temporary file + replace), so a job killed mid-write
    cannot leave a half-written checkpoint for the next one to read.
    """
    _, path = resolve_paths(load=False, save=path)
    if not path:
        return None
    dm = _to_numpy(dm)
    if dm.shape[-2:] != (int(cell.nao),) * 2:
        raise ValueError(f"scf_chk: dm shape {dm.shape} does not end in "
                         f"(nao, nao) = ({cell.nao}, {cell.nao})")
    meta = dict(signature(cell), format=FORMAT, dm_shape=list(dm.shape),
                e_tot=(None if e_tot is None else float(e_tot)), xc=xc,
                ke_cutoff=float(getattr(cell, "ke_cutoff", 0.0) or 0.0),
                mesh=[int(m) for m in cell.mesh], written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, dm=np.ascontiguousarray(dm, dtype=np.float64),
                     coords=np.asarray(cell.atom_coords(), dtype=np.float64),
                     lattice=np.asarray(cell.lattice_vectors(), dtype=np.float64),
                     meta=np.asarray(json.dumps(meta)))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    if verbose:
        _log(f"wrote {path} (dm {tuple(dm.shape)}"
             + (f", E={float(e_tot):.8f} Ha" if e_tot is not None else "") + ")")
    return path


def read(path):
    """``(dm, coords, lattice, meta)`` of a checkpoint file."""
    with np.load(path, allow_pickle=False) as z:
        return (z["dm"], z["coords"], z["lattice"],
                json.loads(str(z["meta"][()])))


def load_dm0(cell, path=None, max_disp_bohr=None, verbose=True):
    """Initial-guess density for ``cell``, or None if there is nothing usable.

    Never raises on a missing, unreadable or mismatched checkpoint -- an SCF
    that cannot reuse a guess should still run, just from scratch.
    """
    path, _ = resolve_paths(load=path, save=False)
    if not path:
        return None
    if not os.path.exists(path):
        if verbose:
            _log(f"no checkpoint at {path}; starting from the default guess")
        return None
    try:
        dm, coords, lattice, meta = read(path)
    except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not kill the run
        _log(f"cannot read {path} ({type(exc).__name__}: {exc}); "
             "starting from the default guess")
        return None

    bad = _mismatch(meta, signature(cell))
    if bad is None and not np.allclose(lattice, cell.lattice_vectors(), atol=1e-9, rtol=0):
        bad = "lattice vectors differ"
    if bad is not None:
        _log(f"ignoring {path}: {bad}")
        return None

    disp = np.asarray(cell.atom_coords()) - coords
    dmax = float(np.abs(disp).max()) if disp.size else 0.0
    if max_disp_bohr is not None and dmax > float(max_disp_bohr):
        _log(f"ignoring {path}: max displacement {dmax * BOHR:.3f} A exceeds "
             f"{float(max_disp_bohr) * BOHR:.3f} A")
        return None
    if verbose:
        e = meta.get("e_tot")
        _log(f"initial guess from {path}: max displacement {dmax * BOHR:.4f} A"
             + (f", checkpoint E={e:.8f} Ha" if e is not None else ""))
    return dm


def run_scf(kg, cell, load=None, save=None, max_disp_bohr=None, verbose=True, **kw):
    """``kg.kernel()`` with a checkpointed initial guess, saving on convergence.

    Works for any pyscf/gpu4pyscf mean-field object: the density matrix is
    stored in whatever shape ``make_rdm1`` produced, so the k-point and spin
    axes of KRKS / KUKS need no special handling here.  Returns ``kg``.
    """
    dm0 = load_dm0(cell, path=load, max_disp_bohr=max_disp_bohr, verbose=verbose)
    if dm0 is not None and type(kg).__module__.startswith("gpu4pyscf"):
        import cupy as cp
        dm0 = cp.asarray(dm0)
    kg.kernel(dm0=dm0, **kw)
    if getattr(kg, "converged", False):
        save_dm(cell, kg.make_rdm1(), path=save, e_tot=float(kg.e_tot),
                xc=getattr(kg, "xc", None), verbose=verbose)
    elif verbose:
        _log("SCF not converged; checkpoint not written")
    return kg


__all__ = ["run_scf", "load_dm0", "save_dm", "read", "resolve_paths", "signature",
           "ENV_BOTH", "ENV_IN", "ENV_OUT"]
