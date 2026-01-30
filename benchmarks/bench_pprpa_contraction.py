"""
Benchmark script for _pprpa_contraction comparing optimized vs reference implementation.
"""

import numpy as np
from benchmarks.benchmark_utils import BenchmarkRunner, setup_pprpa, get_benzene_mol
from lib_pprpa import pprpa_davidson
from benchmarks import references as ref_implementations


def benchmark_pprpa_contraction():
    # Setup test case (Benzene/cc-pVDZ)
    mol = get_benzene_mol(basis='cc-pvdz')
    mf, pp, xy = setup_pprpa(mol, use_df=True, mult='s')
    
    # Generate random trial vectors for testing
    n_vecs = 20
    tri_vec = np.random.random((n_vecs, pp.full_dim))
    
    runner = BenchmarkRunner()
    results = runner.run(
        opt_func=pprpa_davidson._pprpa_contraction,
        ref_func=ref_implementations._pprpa_contraction_ref,
        args=(pp, tri_vec),
        n_iterations=5,
        func_name="_pprpa_contraction"
    )

    print(f"\n\nFor optimizations.md:")
    print(f"| Benzene/cc-pVDZ (20 vecs) | {results.reference_avg:.3f}s | {results.optimized_avg:.3f}s | {results.speedup:.2f}x |")


if __name__ == "__main__":
    benchmark_pprpa_contraction()
