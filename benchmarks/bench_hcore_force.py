"""Density-space hcore force vs the per-atom ``hcore_generator``, at production scale.

Both paths compute the same thing --
``sum_atoms einsum('kxij,kji->x', hcore_deriv(ia), T)`` -- so this script runs
them on the same cell and density and reports the difference and the wall time.
The density only has to be a plausible symmetric AO matrix for that comparison
to be meaningful, so the default skips the SCF and uses the initial guess;
``--scf`` runs the real thing when a physical force is wanted alongside.

    python benchmarks/bench_hcore_force.py <geom.xyz> --ke 600 --charge -3
    python benchmarks/bench_hcore_force.py <geom.xyz> --ke 300 --only density

``LIB_PPRPA_GPUS=2`` splits both paths over two cards, as production does.
"""
from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import numpy as np
import cupy as cp
from ase.io import read
from pyscf.data.nist import BOHR
from pyscf.pbc import gto
from gpu4pyscf.pbc import dft as gdft

from lib_pprpa.gpu_multi import default_group, describe_devices
from lib_pprpa.grad.gpu_hcore_force import hcore_force
from lib_pprpa.grad.pprpa_gamma_gpu import _hcore_force

GAMMA = np.zeros((1, 3))


def build_cell(path, ke, basis, pseudo, charge):
    at = read(path)
    cell = gto.Cell()
    cell.atom = [(s, c) for s, c in zip(at.get_chemical_symbols(),
                                        np.asarray(at.get_positions()) / BOHR)]
    cell.a = np.asarray(at.cell) / BOHR
    cell.unit = "Bohr"
    cell.basis = basis
    cell.pseudo = pseudo
    cell.charge = charge
    cell.spin = 0
    cell.ke_cutoff = ke
    cell.verbose = 0
    cell.build()
    return cell


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("geometry")
    p.add_argument("--ke", type=float, default=600.0)
    p.add_argument("--basis", default="gth-dzvp")
    p.add_argument("--pseudo", default="gth-pbe")
    p.add_argument("--charge", type=int, default=-3)
    p.add_argument("--xc", default="pbe")
    p.add_argument("--scf", action="store_true",
                   help="converge the SCF instead of using the initial guess")
    p.add_argument("--only", choices=("both", "dense", "density"), default="both")
    p.add_argument("--dense-sample", type=int, default=0, metavar="N",
                   help="time the per-atom reference on N atoms only and extrapolate "
                        "to all natm.  The point is a SAME-CARD baseline: the recorded "
                        "NV216 figures (24 min on one B200, 11.5 on two) are B200 "
                        "numbers, and an RTX PRO 6000 Blackwell has ~1/64 the fp64 "
                        "rate, so a cross-card ratio means nothing.  The sampled atoms "
                        "are also compared against the density path element-wise.")
    args = p.parse_args(argv)

    cell = build_cell(args.geometry, args.ke, args.basis, args.pseudo, args.charge)
    group = default_group()
    print(describe_devices(group.devices), flush=True)
    print(f"[bench] {cell.natm} atoms, nao={cell.nao}, "
          f"mesh={[int(m) for m in cell.mesh]}, "
          f"ke={args.ke}, {group.nslots} GPU slot(s)", flush=True)

    kg = gdft.KRKS(cell, kpts=GAMMA, xc=args.xc)
    kg.exxdiv = None
    t0 = time.perf_counter()
    if args.scf:
        kg.conv_tol = 1e-9
        kg.kernel()
        assert kg.converged, "SCF did not converge"
        dm = kg.make_rdm1()
        what = f"converged density (E={float(kg.e_tot):.8f} Ha)"
    else:
        dm = kg.get_init_guess()
        what = "initial-guess density"
    T = np.asarray(cp.asnumpy(dm if not hasattr(dm, "get") else dm.get()))
    if T.ndim == 3:
        T = T[0]
    T = np.ascontiguousarray((T + T.T).real * 0.5, dtype=np.float64)
    print(f"[bench] {what} in {time.perf_counter() - t0:.1f} s", flush=True)

    mo = np.eye(cell.nao)
    shim = SimpleNamespace(xc=args.xc, exxdiv=None, mo_coeff=mo,
                           mo_energy=np.zeros(cell.nao), mo_occ=np.zeros(cell.nao))
    kg = None
    cp.get_default_memory_pool().free_all_blocks()

    # The density path runs FIRST: it is the cheap one, and the per-atom
    # reference can take hours at NV216 on a card with weak fp64.  If the job
    # runs out of walltime the number we came for is already in the log.
    out = {}
    if args.only in ("both", "density"):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        out["density"], stats = hcore_force(cell, GAMMA, T, group=group)
        cp.cuda.Device().synchronize()
        out["density_seconds"] = time.perf_counter() - t0
        print(f"[bench] density-space:            {out['density_seconds']:.1f} s "
              f"(rho {stats['rho_seconds']:.1f}, G-space {stats['reduce_seconds']:.1f}, "
              f"AO-derivative {stats['h1_seconds']:.1f})", flush=True)

    if args.only in ("both", "dense"):
        from lib_pprpa.grad.pprpa_gamma_gpu import _make_kmf
        gg = _make_kmf(cell, GAMMA, shim, True).nuc_grad_method()
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        print(f"[bench] per-atom hcore_generator: {cell.natm} atoms, starting...", flush=True)
        out["dense"], _ = _hcore_force(gg, cell, GAMMA, shim, True, T, group)
        cp.cuda.Device().synchronize()
        out["dense_seconds"] = time.perf_counter() - t0
        gg = None
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[bench] per-atom hcore_generator: {out['dense_seconds']:.1f} s "
              f"({out['dense_seconds'] / cell.natm:.2f} s/atom)", flush=True)

    if args.dense_sample:
        from gpu4pyscf.pbc.grad import krhf as krhf_g
        from lib_pprpa.grad.pprpa_gamma_gpu import _make_kmf
        n = min(int(args.dense_sample), cell.natm)
        gg = _make_kmf(cell, GAMMA, shim, True).nuc_grad_method()
        deriv = krhf_g.hcore_generator(gg, cell, GAMMA)   # builds get_hcore once
        Tg = cp.asarray(T)[None]
        cp.cuda.Device().synchronize()
        _ = cp.einsum('kxij,kji->x', deriv(0), Tg).real   # warm up, not timed
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        sample = np.stack([cp.asnumpy(cp.einsum('kxij,kji->x', deriv(ia), Tg).real)
                           for ia in range(n)])
        cp.cuda.Device().synchronize()
        per_atom = (time.perf_counter() - t0) / n
        full = per_atom * cell.natm
        deriv = gg = Tg = None
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[bench] per-atom reference, {n}/{cell.natm} atoms sampled: "
              f"{per_atom:.2f} s/atom -> {full:.0f} s ({full/60:.1f} min) for all "
              f"{cell.natm}, on THIS card", flush=True)
        if "density" in out:
            a, b = sample, out["density"][:n]
            scale = max(np.abs(a).max(), 1e-30)
            d = np.abs(a - b).max()
            print(f"[bench] sampled atoms vs density path: max abs diff {d:.3e} "
                  f"({d / scale:.2e} relative)", flush=True)
            print(f"[bench] extrapolated speedup = "
                  f"{full / out['density_seconds']:.0f}x", flush=True)

    if args.only == "both":
        a, b = out["dense"], out["density"]
        scale = max(np.abs(a).max(), 1e-30)
        d = np.abs(a - b)
        print(f"[bench] max|de| = {scale:.6e} a.u.")
        print(f"[bench] max abs diff = {d.max():.3e}  ({d.max() / scale:.2e} relative)")
        print(f"[bench] rms abs diff = {np.sqrt((d ** 2).mean()):.3e}")
        worst = int(np.unravel_index(d.argmax(), d.shape)[0])
        print(f"[bench] worst atom {worst}: dense {a[worst]}, density {b[worst]}")
        print(f"[bench] speedup = {out['dense_seconds'] / out['density_seconds']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
