"""
Synthesis Benchmarking Module

A simple, extensible framework for benchmarking quantum circuit synthesis compilers.
Each compiler takes a ProblemInstance and produces a phase-flip oracle circuit.

Key Components:
- SynthesisCompiler: Abstract interface for synthesis compilers
- SynthesisTrial: Stores synthesis benchmark results
- SynthesisBenchmark: Orchestrates the benchmarking process
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.types import JSON
from sqlalchemy.orm import mapped_column
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import IBMBackend

from ..core import BaseTrial, BenchmarkDatabase, BaseProblem

logger = logging.getLogger("benchmarklib.compiler")


class SynthesisCompiler(ABC):
    """
    Abstract base class for quantum circuit synthesis compilers.

    Each compiler implementation must:
    1. Take a ProblemInstance as input
    2. Generate the appropriate classical function representation
    3. Synthesize a PHASE-FLIP ORACLE quantum circuit

    The phase-flip oracle should flip the phase of computational basis states
    that satisfy the problem constraints (i.e., |x⟩ → -|x⟩ for solutions).

    Example Implementation:
        class XAGCompiler(SynthesisCompiler):
            def __init__(self, optimize_xag=True):
                self.optimize_xag = optimize_xag

            @property
            def name(self) -> str:
                return "XAG_OPTIMIZED" if self.optimize_xag else "XAG"

            def compile(self, problem: ProblemInstance, **kwargs) -> QuantumCircuit:
                # Your synthesis logic here
                oracle = synthesize_with_xag(problem, self.optimize_xag, **kwargs)
                return oracle
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this compiler.

        Used for tracking results and analysis. Should be descriptive
        and consistent across runs (e.g., "XAG", "CLASSICAL_FUNCTION", "QISKIT_SYNTH").

        Returns:
            String identifier for the compiler
        """
        pass

    @abstractmethod
    def compile(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
        """
        Synthesize a phase-flip oracle for the given problem instance.

        This method should:
        1. Extract problem parameters (size, constraints, etc.)
        2. Generate/retrieve the classical function representation
        3. Apply synthesis algorithms to create the quantum circuit
        4. Return a PHASE-FLIP oracle (not bit-flip!)

        The oracle should implement: |x⟩|y⟩ → (-1)^f(x)|x⟩|y⟩
        where f(x) = 1 if x is a solution to the problem.

        Args:
            problem: ProblemInstance to compile into an oracle
            **kwargs: Problem-specific parameters (e.g., clique_size for clique problems)
                     These are the same kwargs passed to ProblemInstance.oracle()

        Returns:
            QuantumCircuit implementing the phase-flip oracle

        Raises:
            NotImplementedError: If the compiler doesn't support this problem type
            ValueError: If required kwargs are missing or invalid
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def get_config(self) -> Dict[str, Any]:
        return {}


@dataclass
class SynthesisResult:
    """
    Container for synthesis compilation results.

    Stores all metrics from a single synthesis run, including circuit
    characteristics and compilation performance.
    """

    compiler_name: str
    success: bool
    synthesis_time: float  # Total end-to-end time in seconds
    error_message: Optional[str] = None

    # Circuit metrics (None if compilation failed)
    num_qubits: Optional[int] = None
    circuit_depth: Optional[int] = None
    circuit_size: Optional[int] = None  # Total gate count
    cx_count: Optional[int] = None
    single_qubit_count: Optional[int] = None

    # Additional metrics can be added here
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class SynthesisTrial(BaseTrial):
    """
    Trial for synthesis benchmarking.

    Extends BaseTrial to store synthesis-specific metrics while maintaining
    compatibility with the existing database infrastructure.
    """

    __abstract__ = True

    synthesis_result: Optional[SynthesisResult] = mapped_column(MutableDict.as_mutable(JSON), nullable=True)

    def __init__(
        self,
        problem_instance: BaseProblem,
        job_id: Optional[str] = None,
        job_pub_idx: int = 0,
        counts: Optional[Dict[str, Any]] = None,
        simulation_counts: Optional[Dict[str, int]] = None,
        trial_id: Optional[int] = None,
        created_at: Optional[str] = None,
        # Additional synthesis-specific parameters (only used for new trials)
        compiler_name: Optional[str] = None,
        synthesis_result: Optional[SynthesisResult] = None,
        **trial_params,
    ):
        """
        Initialize a synthesis benchmark trial.

        This constructor handles two cases:
        1. Loading from database: All BaseTrial parameters are provided
        2. Creating new trial: compiler_name and synthesis_result are provided

        Args:
            instance_id: ID of the problem instance
            job_id: "SYNTHESIS_<compiler_name>" format
            job_pub_idx: Always 0 for synthesis
            counts: Dict containing synthesis metrics (from DB) or None (new)
            simulation_counts: Always None for synthesis
            trial_id: Database ID
            created_at: Timestamp
            compiler_name: Compiler name (only for new trials)
            synthesis_result: Synthesis results (only for new trials)
            **trial_params: Additional parameters like clique_size
        """
        # Case 1: Loading from database
        if counts is not None and job_id and job_id.startswith("SYNTHESIS_"):
            self.compiler_name = job_id.replace("SYNTHESIS_", "")

            # Reconstruct SynthesisResult from stored metrics
            self.synthesis_result = SynthesisResult(
                compiler_name=self.compiler_name,
                success=counts.get("success", False),
                synthesis_time=counts.get("synthesis_time", 0.0),
                error_message=counts.get("error_message"),
                num_qubits=counts.get("num_qubits"),
                circuit_depth=counts.get("circuit_depth"),
                circuit_size=counts.get("circuit_size"),
                cx_count=counts.get("cx_count"),
                single_qubit_count=counts.get("single_qubit_count"),
                extra_metrics={
                    k: v
                    for k, v in counts.items()
                    if k
                    not in [
                        "success",
                        "synthesis_time",
                        "error_message",
                        "num_qubits",
                        "circuit_depth",
                        "circuit_size",
                        "cx_count",
                        "single_qubit_count",
                    ]
                },
            )

        # Case 2: Creating new trial
        elif compiler_name and synthesis_result:
            self.compiler_name = compiler_name
            self.synthesis_result = synthesis_result

            # Prepare data for storage
            job_id = f"SYNTHESIS_{compiler_name}"
            counts = {
                "success": synthesis_result.success,
                "synthesis_time": synthesis_result.synthesis_time,
                "error_message": synthesis_result.error_message,
                "num_qubits": synthesis_result.num_qubits,
                "circuit_depth": synthesis_result.circuit_depth,
                "circuit_size": synthesis_result.circuit_size,
                "cx_count": synthesis_result.cx_count,
                "single_qubit_count": synthesis_result.single_qubit_count,
                **synthesis_result.extra_metrics,
            }

        else:
            raise ValueError(
                "SynthesisTrial requires either database parameters "
                "(counts and job_id) or new trial parameters "
                "(compiler_name and synthesis_result)"
            )

        # Call parent constructor with standard parameters
        super().__init__(
            problem_instance=problem_instance,
            compiler_name=compiler_name,
            job_id=job_id,
            job_pub_idx=job_pub_idx,
            counts=counts,
            simulation_counts=simulation_counts,
            trial_id=trial_id,
            created_at=created_at,
            **trial_params,
        )

    def calculate_success_rate(
        self, db_manager: Optional[BenchmarkDatabase] = None
    ) -> float:
        """For synthesis trials, return 1.0 if successful, 0.0 if failed."""
        return 1.0 if self.synthesis_result.success else 0.0

    def calculate_expected_success_rate(
        self, db_manager: Optional[BenchmarkDatabase] = None
    ) -> float:
        """For synthesis trials, expected success rate is always 1.0."""
        return 1.0


class SynthesisBenchmark:
    """
    Orchestrates synthesis benchmarking across multiple compilers and problems.

    This class provides a simple interface for:
    1. Running multiple compilers on the same problem instances
    2. Collecting standardized metrics
    3. Storing results in the database
    4. Comparing compiler performance

    Example Usage:
        # Set up compilers
        compilers = [
            XAGCompiler(optimize=True),
            ClassicalFunctionCompiler(),
            MyNewCompiler()
        ]

        # Create benchmark
        benchmark = SynthesisBenchmark(db_manager, compilers)

        # Run on specific problems
        problems = db_manager.find_problem_instances(size_filters={"num_vars": 5})
        results = benchmark.run_benchmarks(problems, clique_size=3)

        # Analyze results
        benchmark.print_summary(results)
    """

    def __init__(
        self,
        db_manager: BenchmarkDatabase,
        compilers: List[SynthesisCompiler],
        save_to_db: bool = True,
        backend: Optional[IBMBackend] = None,
    ):
        """
        Initialize synthesis benchmark.

        Args:
            db_manager: Database manager for saving results
            compilers: List of compiler implementations to benchmark
            save_to_db: Whether to save trials to the database
            backend: Optional backend for transpilation metrics
        """
        self.db_manager = db_manager
        self.compilers = compilers
        self.save_to_db = save_to_db
        self.backend = backend

        logger.info(f"SynthesisBenchmark initialized with {len(compilers)} compilers")

    def benchmark_single(
        self, compiler: SynthesisCompiler, problem: BaseProblem, **kwargs
    ) -> SynthesisResult:
        """
        Run a single compiler on a single problem instance.

        Args:
            compiler: Compiler to use
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters

        Returns:
            SynthesisResult with compilation metrics
        """
        logger.debug(f"Benchmarking {compiler.name} on {problem}")

        # Initialize result
        result = SynthesisResult(
            compiler_name=compiler.name, success=False, synthesis_time=0.0
        )

        try:
            # Time the compilation
            start_time = time.time()
            oracle = compiler.compile(problem, **kwargs)
            end_time = time.time()

            result.synthesis_time = end_time - start_time
            result.success = True

            # Extract circuit metrics
            result.num_qubits = oracle.num_qubits

            # Optionally transpile for more accurate metrics
            if self.backend:
                oracle = transpile(oracle, self.backend, optimization_level=3)

            result.circuit_depth = oracle.depth()
            result.circuit_size = oracle.size()

            # Count gate types
            ops_count = oracle.count_ops()
            result.cx_count = ops_count.get("cx", 0) + ops_count.get("cnot", 0)

            # Count single-qubit gates (exclude barriers, measurements)
            excluded_ops = {"cx", "cnot", "barrier", "measure"}
            result.single_qubit_count = sum(
                count for op, count in ops_count.items() if op not in excluded_ops
            )

            logger.debug(
                f"Success: {result.num_qubits} qubits, "
                f"depth {result.circuit_depth}, "
                f"time {result.synthesis_time:.3f}s"
            )

        except Exception as e:
            result.error_message = str(e)
            logger.warning(f"Compilation failed: {e}")

        return result

    def run_benchmarks(
        self, problems: List[BaseProblem], skip_existing: bool = True, **kwargs
    ) -> Dict[str, List[SynthesisResult]]:
        """
        Run all compilers on all problem instances.

        Args:
            problems: List of problem instances to compile
            skip_existing: Skip if trial already exists in database
            **kwargs: Problem-specific parameters passed to compilers

        Returns:
            Dictionary mapping compiler names to lists of results
        """
        results = {compiler.name: [] for compiler in self.compilers}

        total_runs = len(problems) * len(self.compilers)
        current_run = 0

        for problem in problems:
            logger.info(f"Processing problem {problem.instance_id}: {problem}")

            for compiler in self.compilers:
                current_run += 1
                logger.info(
                    f"  [{current_run}/{total_runs}] Running {compiler.name}..."
                )

                # Check for existing trial
                if skip_existing and self.save_to_db:
                    existing = self.db_manager.find_trials(
                        instance_id=problem.instance_id,
                        job_id=f"SYNTHESIS_{compiler.name}",
                        trial_params=kwargs,
                    )
                    if existing:
                        logger.info("    Skipping (already exists)")
                        continue

                # Run benchmark
                result = self.benchmark_single(compiler, problem, **kwargs)
                results[compiler.name].append(result)

                # Save to database
                if self.save_to_db:
                    trial = SynthesisTrial(
                        problem_instance=problem,
                        compiler_name=compiler.name,
                        synthesis_result=result,
                        **kwargs,
                    )
                    self.db_manager.save_trial(trial)
                    logger.debug(f"    Saved trial {trial.trial_id}")

        return results

    def print_summary(self, results: Dict[str, List[SynthesisResult]]) -> None:
        """
        Print a summary of benchmark results.

        Args:
            results: Dictionary from run_benchmarks()
        """
        print("\n" + "=" * 60)
        print("SYNTHESIS BENCHMARK SUMMARY")
        print("=" * 60)

        for compiler_name, compiler_results in results.items():
            if not compiler_results:
                continue

            print(f"\n{compiler_name}:")
            print("-" * 40)

            # Success rate
            successes = sum(1 for r in compiler_results if r.success)
            success_rate = successes / len(compiler_results) * 100
            print(
                f"  Success Rate: {successes}/{len(compiler_results)} ({success_rate:.1f}%)"
            )

            # Only analyze successful compilations
            successful = [r for r in compiler_results if r.success]
            if successful:
                # Timing
                times = [r.synthesis_time for r in successful]
                print(f"  Synthesis Time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")

                # Circuit metrics
                qubits = [r.num_qubits for r in successful]
                depths = [r.circuit_depth for r in successful]
                sizes = [r.circuit_size for r in successful]
                cx_counts = [r.cx_count for r in successful]

                print(f"  Avg Qubits: {np.mean(qubits):.1f} ± {np.std(qubits):.1f}")
                print(f"  Avg Depth: {np.mean(depths):.1f} ± {np.std(depths):.1f}")
                print(f"  Avg Gates: {np.mean(sizes):.1f} ± {np.std(sizes):.1f}")
                print(
                    f"  Avg CX Gates: {np.mean(cx_counts):.1f} ± {np.std(cx_counts):.1f}"
                )

            # Failures
            failures = [r for r in compiler_results if not r.success]
            if failures:
                print(f"  Failures: {len(failures)}")
                # Show first error as example
                if failures[0].error_message:
                    print(f"    Example error: {failures[0].error_message[:100]}...")

        print("\n" + "=" * 60)


def compare_compilers(
    db_manager: BenchmarkDatabase,
    compiler_names: List[str],
    problem_filters: Optional[Dict[str, Any]] = None,
    **trial_params,
) -> None:
    """
    Compare synthesis results from different compilers.

    This is a utility function for analyzing stored results.

    Args:
        db_manager: Database with stored trials
        compiler_names: List of compiler names to compare
        problem_filters: Filters for selecting problems
        **trial_params: Trial parameters to filter by
    """
    import pandas as pd

    # Get relevant problems
    problems = db_manager.find_problem_instances(size_filters=problem_filters)

    data = []
    for problem in problems:
        problem_size = problem.get_problem_size()

        for compiler_name in compiler_names:
            # Find trials for this compiler and problem
            trials = db_manager.find_trials(
                instance_id=problem.instance_id,
                job_id=f"SYNTHESIS_{compiler_name}",
                trial_params=trial_params,
            )

            for trial in trials:
                if trial.counts and isinstance(trial.counts, dict):
                    metrics = trial.counts  # Synthesis metrics stored here
                    data.append(
                        {
                            "compiler": compiler_name,
                            "problem_id": problem.instance_id,
                            **problem_size,
                            **metrics,
                            **trial_params,
                        }
                    )

    if data:
        df = pd.DataFrame(data)

        # Group by compiler and show averages
        print("\nCompiler Comparison:")
        print("=" * 60)

        metrics_to_compare = [
            "synthesis_time",
            "num_qubits",
            "circuit_depth",
            "cx_count",
        ]

        for metric in metrics_to_compare:
            if metric in df.columns:
                print(f"\n{metric}:")
                summary = df.groupby("compiler")[metric].agg(
                    ["mean", "std", "min", "max"]
                )
                print(summary)
    else:
        print("No data found for comparison")

