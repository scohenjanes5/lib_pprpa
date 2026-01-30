"""
Benchmark script comparing optimized vs reference implementations of make_rdm1_relaxed_rhf_pprpa.
"""

from benchmarks.benchmark_utils import BenchmarkRunner, setup_pprpa, get_benzene_mol
from lib_pprpa.grad import pprpa as pprpa_grad_mod
from benchmarks import references as ref_implementations


def benchmark_make_rdm1():
    # Setup test case (Benzene/cc-pVDZ)
    mol = get_benzene_mol(basis='cc-pvdz')
    mf, pp, xy = setup_pprpa(mol, use_df=True, mult='s')
    
    runner = BenchmarkRunner()
    results = runner.run(
        opt_func=pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa,
        ref_func=ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref,
        args=(pp, mf),
        kwargs={'xy': xy, 'mult': 's'},
        n_iterations=5,
        func_name="make_rdm1_relaxed_rhf_pprpa"
    )

    print(f"\n\nFor optimizations.md:")
    print(f"| Benzene/cc-pVDZ | {results.reference_avg:.3f}s | {results.optimized_avg:.3f}s | {results.speedup:.2f}x |")


if __name__ == "__main__":
    benchmark_make_rdm1()
