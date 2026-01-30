"""
Generic benchmark framework for comparing optimized vs reference implementations.

This module provides utilities for benchmarking ppRPA functions.

Usage:
    from benchmarks.benchmark_utils import BenchmarkRunner, get_benzene_mol, get_water_mol
    
    runner = BenchmarkRunner()
    results = runner.run(opt_func, ref_func, args, kwargs, n_iterations=5)
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional, Tuple
from pyscf import gto, dft, scf


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    optimized_times: List[float]
    reference_times: List[float]
    optimized_avg: float
    optimized_std: float
    reference_avg: float
    reference_std: float
    speedup: float
    correct: bool
    opt_result: Any = None
    ref_result: Any = None


def get_water_mol(basis='6-31g'):
    """Create a water molecule."""
    mol = gto.Mole()
    mol.atom = '''
        O   0.000000   0.000000   0.000000
        H   0.000000   0.757000   0.587000
        H   0.000000  -0.757000   0.587000
    '''
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    return mol


def get_benzene_mol(basis='cc-pvdz'):
    """Create a benzene molecule."""
    mol = gto.Mole()
    mol.atom = '''
        C   0.000000   1.396000   0.000000
        C   1.209300   0.698000   0.000000
        C   1.209300  -0.698000   0.000000
        C   0.000000  -1.396000   0.000000
        C  -1.209300  -0.698000   0.000000
        C  -1.209300   0.698000   0.000000
        H   0.000000   2.479000   0.000000
        H   2.147000   1.240000   0.000000
        H   2.147000  -1.240000   0.000000
        H   0.000000  -2.479000   0.000000
        H  -2.147000  -1.240000   0.000000
        H  -2.147000   1.240000   0.000000
    '''
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    return mol


def get_naphthalene_mol(basis='cc-pvdz'):
    """Create a naphthalene molecule (larger test case)."""
    mol = gto.Mole()
    mol.atom = '''
        C   0.000000   0.000000   1.405000
        C   0.000000   1.220000   0.718000
        C   0.000000   1.220000  -0.718000
        C   0.000000   0.000000  -1.405000
        C   0.000000  -1.220000  -0.718000
        C   0.000000  -1.220000   0.718000
        C   0.000000   2.469000   1.399000
        C   0.000000   2.469000  -1.399000
        C   0.000000   3.651000   0.718000
        C   0.000000   3.651000  -0.718000
        H   0.000000   0.000000   2.496000
        H   0.000000   0.000000  -2.496000
        H   0.000000  -2.168000   1.254000
        H   0.000000  -2.168000  -1.254000
        H   0.000000   2.469000   2.490000
        H   0.000000   2.469000  -2.490000
        H   0.000000   4.599000   1.254000
        H   0.000000   4.599000  -1.254000
    '''
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    return mol


class BenchmarkRunner:
    """Runner for comparing optimized vs reference implementations."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def compare_results(self, opt_result: Any, ref_result: Any, 
                       atol: float = 1e-8) -> bool:
        """Compare results from optimized and reference implementations."""
        if isinstance(opt_result, tuple):
            return all(
                np.allclose(o, r, atol=atol) 
                for o, r in zip(opt_result, ref_result)
            )
        elif isinstance(opt_result, np.ndarray):
            return np.allclose(opt_result, ref_result, atol=atol)
        else:
            return opt_result == ref_result
    
    def run(
        self,
        opt_func: Callable,
        ref_func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        n_iterations: int = 5,
        warmup: int = 1,
        atol: float = 1e-8,
        func_name: str = "function"
    ) -> BenchmarkResult:
        """
        Run benchmark comparing optimized vs reference implementation.
        
        Args:
            opt_func: Optimized function to benchmark
            ref_func: Reference function to compare against
            args: Positional arguments to pass to both functions
            kwargs: Keyword arguments to pass to both functions
            n_iterations: Number of timed iterations
            warmup: Number of warmup iterations (not timed)
            atol: Absolute tolerance for correctness check
            func_name: Name for logging
            
        Returns:
            BenchmarkResult with timing and correctness data
        """
        kwargs = kwargs or {}
        
        self.log(f"\n{'='*60}")
        self.log(f"Benchmarking {func_name}")
        self.log(f"{'='*60}\n")
        
        # Warmup
        self.log(f"Warm-up runs ({warmup} iterations)...")
        for _ in range(warmup):
            _ = opt_func(*args, **kwargs)
            _ = ref_func(*args, **kwargs)
        
        # Benchmark optimized version
        self.log(f"\nBenchmarking OPTIMIZED version ({n_iterations} iterations)...")
        times_opt = []
        opt_result = None
        for i in range(n_iterations):
            start = time.perf_counter()
            opt_result = opt_func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times_opt.append(elapsed)
            self.log(f"  Iteration {i+1}: {elapsed:.4f}s")
        
        # Benchmark reference version
        self.log(f"\nBenchmarking REFERENCE version ({n_iterations} iterations)...")
        times_ref = []
        ref_result = None
        for i in range(n_iterations):
            start = time.perf_counter()
            ref_result = ref_func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times_ref.append(elapsed)
            self.log(f"  Iteration {i+1}: {elapsed:.4f}s")
        
        # Verify correctness
        self.log("\nVerifying correctness...")
        correct = self.compare_results(opt_result, ref_result, atol=atol)
        self.log(f"  Results match: {'✓' if correct else '✗'}")
        
        # Calculate statistics
        avg_opt = np.mean(times_opt)
        std_opt = np.std(times_opt)
        avg_ref = np.mean(times_ref)
        std_ref = np.std(times_ref)
        speedup = avg_ref / avg_opt if avg_opt > 0 else float('inf')
        
        # Report results
        self.log(f"\n{'='*60}")
        self.log("BENCHMARK RESULTS")
        self.log(f"{'='*60}")
        self.log(f"OPTIMIZED: {avg_opt:.4f}s ± {std_opt:.4f}s per call")
        self.log(f"REFERENCE: {avg_ref:.4f}s ± {std_ref:.4f}s per call")
        self.log(f"\nSpeedup: {speedup:.2f}x")
        
        if speedup > 1.05:
            self.log(f"Optimized version is {speedup:.2f}x FASTER")
        elif speedup < 0.95:
            self.log(f"Optimized version is {1/speedup:.2f}x SLOWER")
        else:
            self.log("No significant difference")
        
        return BenchmarkResult(
            optimized_times=times_opt,
            reference_times=times_ref,
            optimized_avg=avg_opt,
            optimized_std=std_opt,
            reference_avg=avg_ref,
            reference_std=std_ref,
            speedup=speedup,
            correct=correct,
            opt_result=opt_result,
            ref_result=ref_result
        )


def setup_pprpa(mol, xc='B3LYP', use_df=True, mult='s'):
    """
    Set up ppRPA calculation for a molecule.
    
    Returns:
        tuple: (mf, pprpa_obj, xy)
    """
    from lib_pprpa import pyscf_util
    from lib_pprpa.grad.ase_utils import pprpaobj
    
    # Run DFT
    if xc:
        mf = dft.RKS(mol, xc=xc)
    else:
        mf = scf.RHF(mol)
    
    if use_df:
        mf = mf.density_fit()
    mf.kernel()
    
    # Set up ppRPA
    nocc, mo_energy, Lpq = pyscf_util.get_pyscf_input_mol(mf)
    pp = pprpaobj(mf, 'pp', Lpq=Lpq, nroot=1)
    pp.kernel(mult)
    
    xy = pp.xy_s[0] if mult == 's' else pp.xy_t[0]
    
    return mf, pp, xy
