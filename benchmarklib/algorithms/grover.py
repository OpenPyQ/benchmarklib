"""
Grover Algorithm Execution Library - Batch Mode

A unified library for running Grover's algorithm with efficient batch processing
that integrates seamlessly with the quantum benchmarking database.

Key Features:
- Problem-agnostic Grover circuit construction
- Integration with BenchmarkDatabase
- Efficient batch processing with qiskit_ibm_runtime.Batch
- Automatic job grouping by problem parameters
- Both simulation and hardware execution

"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import qiskit
from qiskit import transpile
from qiskit.circuit.library import grover_operator, QFT
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# IBM Quantum imports
from qiskit_ibm_runtime import Batch, QiskitRuntimeService, SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options import dynamical_decoupling_options

from ..core import BaseTrial, BenchmarkDatabase, BaseProblem
from ..compilers import SynthesisCompiler

logger = logging.getLogger("benchmarklib.algorithms.grover")


@dataclass
class GroverConfig:
    """Configuration for Grover algorithm execution."""

    shots: int = 1000
    optimization_level: int = 1
    run_simulation: bool = True
    max_circuit_depth: Optional[int] = None
    timeout_seconds: int = 300
    dynamical_decoupling: bool = False


class GroverRunner:
    """
    Grover algorithm runner with efficient batch processing.

    This class provides batch processing for Grover benchmarks, collecting
    circuits and submitting them in optimally-sized jobs within batch contexts.
    """

    def __init__(
        self,
        db_manager: BenchmarkDatabase,
        service: QiskitRuntimeService,
        backend: Backend,
        config: Optional[GroverConfig] = None,
    ):
        """
        Initialize Grover runner with batch processing support.

        Args:
            db_manager: Database manager for saving results
            service: IBM Quantum service
            backend: Backend for quantum hardware
            config: Execution configuration
        """
        self.db_manager = db_manager
        self.service = service
        self.backend = backend
        self.config = config or GroverConfig()

        # Initialize simulator
        self.simulator = AerSimulator()

        # Batch state management
        self._batch_context = None
        self._batch_circuits = []  # List of (circuit, trial, metadata) tuples
        self._current_compiler = None
        self._batch_job_count = 0

        logger.info(f"GroverRunner initialized for {db_manager.problem_type} problems")

    def build_grover_circuit(
        self, oracle: qiskit.QuantumCircuit, num_vars: int, grover_iterations: int
    ) -> qiskit.QuantumCircuit:
        """
        Alias for build_grover_circuit function.
        """
        return build_grover_circuit(oracle, num_vars, grover_iterations)

    def run_simulation(
        self, circuit: qiskit.QuantumCircuit
    ) -> Optional[Dict[str, int]]:
        """
        Run classical simulation.

        Args:
            circuit: Quantum circuit to simulate

        Returns:
            Measurement counts or None if failed
        """
        if not self.config.run_simulation:
            return None

        try:
            # Transpile for simulator
            qc = transpile(
                circuit,
                self.simulator,
                optimization_level=self.config.optimization_level,
            )

            # Check complexity limits
            if (
                self.config.max_circuit_depth
                and qc.depth() > self.config.max_circuit_depth
            ):
                logger.warning(f"Circuit too deep: {qc.depth()}")
                return None

            # Run simulation
            result = self.simulator.run(qc, shots=self.config.shots).result()
            counts = result.get_counts()

            logger.debug(f"Simulation completed: {len(counts)} unique outcomes")
            return counts

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return None

    def start_batch(self, compiler: SynthesisCompiler) -> None:
        """
        Initialize batch context for a compile type.

        Args:
            compiler: Compilation method for this batch

        Raises:
            ValueError: If batch already active or invalid compiler
        """
        if self._batch_context is not None:
            raise ValueError("Batch already active - call finish_batch() first")

        logger.info(f"Starting batch for {compiler.name}")

        self._batch_context = Batch(backend=self.backend)
        self._batch_context.__enter__()
        self._current_compiler = compiler
        self._batch_circuits = []
        self._batch_job_count = 0

    def run_grover_benchmark(
        self,
        problem_instance: BaseProblem,
        compiler: SynthesisCompiler,
        grover_iterations: int,
        shots: Optional[int] = None,
        save_to_db: bool = True,
        skip_existing: bool = True,
        **oracle_kwargs,
    ) -> BaseTrial:
        """
        Collect Grover benchmark circuit for batch submission.

        This method now COLLECTS circuits rather than immediately submitting them.
        Use submit_job() to actually submit accumulated circuits.

        Args:
            problem_instance: Problem to solve
            compiler: Oracle compilation method
            grover_iterations: Number of Grover iterations
            shots: Number of shots (uses config default if None)
            save_to_db: Whether to save trial to database
            skip_existing: Whether to skip if trial already exists
            **oracle_kwargs: Additional oracle parameters

        Returns:
            Trial object (job_id will be None until submit_job() is called)
        """
        # Validate batch state
        if self._batch_context is None:
            raise ValueError("No active batch - call start_batch() first")

        if compiler.name != self._current_compiler.name:
            raise ValueError(
                f"Compile type {compiler.name} doesn't match current batch {self._current_compiler.name}"
            )

        # Check for existing trial
        if skip_existing:

            existing_trials = self.db_manager.find_trials(
                problem_id=problem_instance.id,
                compiler_name=compiler.name,
                grover_iterations=grover_iterations,
                include_pending=True,
            )

            if existing_trials:
                logger.info(
                    f"Skipping existing trial: Problem {problem_instance.id}, "
                    f"{compiler.name}, {grover_iterations} iterations"
                )
                return existing_trials[0]

        # Handle shots override
        original_shots = None
        if shots:
            original_shots = self.config.shots
            self.config.shots = shots

        try:
            logger.debug(
                f"Collecting circuit for Problem Instance ID: {problem_instance.id}"
            )
            print("COMPILING")

            # Generate oracle
            oracle = compiler.compile(problem_instance, **oracle_kwargs)

            # Determine number of variables
            num_vars = problem_instance.number_of_input_bits()

            # Build Grover circuit
            grover_circuit = self.build_grover_circuit(
                oracle, num_vars, grover_iterations
            )

            # Run simulation
            try:
                simulation_counts = self.run_simulation(grover_circuit)
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                simulation_counts = None

            trial = self.db_manager.trial_class(
                problem=problem_instance,
                compiler_name=compiler.name,
                job_id=None,  # Will be set during submit_job()
                job_pub_idx=None,  # Will be set during submit_job()
                simulation_counts=simulation_counts,
                grover_iterations=grover_iterations,
            )

            # Save trial to database immediately (if requested)
            if save_to_db:
                self.db_manager.save_trial(trial)
                logger.debug(f"Saved trial {trial.id} (pending batch submission)")

            # Add to batch collection
            circuit_metadata = {
                "problem_id": problem_instance.id,
                "compiler_name": compiler.name,
                "grover_iterations": grover_iterations,
                "trial_id": trial.id,
            }

            self._batch_circuits.append((grover_circuit, trial, circuit_metadata))

            logger.debug(
                f"Collected circuit {len(self._batch_circuits)} for batch "
                f"(Problem {problem_instance.id}, {grover_iterations} iterations)"
            )

            return trial

        except Exception as e:
            print(e)
            logger.error(f"Circuit collection failed: {e}")

            # Create failed trial
            trial = self.db_manager.trial_class(
                problem=problem_instance,
                compiler_name=compiler.name,
                grover_iterations=grover_iterations,
                **oracle_kwargs,
            )
            trial.mark_failure()

            if save_to_db:
                self.db_manager.save_trial(trial)

            return trial

        finally:
            # Restore original shots
            if original_shots is not None:
                self.config.shots = original_shots

    def submit_job(self) -> Optional[str]:
        """
        Submit accumulated circuits as a single job within the current batch.

        Returns:
            Job ID if circuits were submitted, None if no circuits to submit

        Raises:
            ValueError: If no active batch or no circuits collected
        """
        if self._batch_context is None:
            raise ValueError("No active batch - call start_batch() first")

        if not self._batch_circuits:
            logger.warning("No circuits to submit")
            return None

        circuit_count = len(self._batch_circuits)
        logger.info(f"Submitting job with {circuit_count} circuits")

        # Issue warning for large jobs
        if circuit_count > 200:
            logger.warning(
                f"Large job size: {circuit_count} circuits. "
                f"Consider splitting into smaller jobs for better performance."
            )

        try:
            # Extract circuits and trials
            circuits = [item[0] for item in self._batch_circuits]
            trials = [item[1] for item in self._batch_circuits]
            metadata_list = [item[2] for item in self._batch_circuits]

            # Transpile circuits
            logger.debug(f"Transpiling {len(circuits)} circuits...")
            transpiled_circuits = transpile(
                circuits,
                backend=self.backend,
                optimization_level=self.config.optimization_level,
            )

            # Submit job within batch context
            sampler = Sampler(mode=self._batch_context)

            # Configure dynamical decoupling options
            sampler.options.dynamical_decoupling.enable = (
                self.config.dynamical_decoupling
            )

            # Submit job
            job = sampler.run(transpiled_circuits, shots=self.config.shots)
            job_id = job.job_id()

            # Update all trials with job information
            for idx, trial in enumerate(trials):
                trial.job_id = job_id
                trial.job_pub_idx = idx
                self.db_manager.save_trial(trial)

            self._batch_job_count += 1

            logger.info(
                f"Submitted job {job_id} with {circuit_count} circuits "
                f"(Job {self._batch_job_count} in batch)"
            )

            # Clear collected circuits for next job
            self._batch_circuits = []

            return job_id

        except Exception as e:
            logger.error(f"Job submission failed: {e}")

            # Mark all trials as failed
            for _, trial, _ in self._batch_circuits:
                trial.mark_failure()
                self.db_manager.save_trial(trial)

            # Clear circuits even on failure
            self._batch_circuits = []

            return None

    def finish_batch(self) -> None:
        """
        Finish the current batch and clean up resources.

        Raises:
            ValueError: If no active batch
        """
        if self._batch_context is None:
            raise ValueError("No active batch to finish")

        # Submit any remaining circuits
        if self._batch_circuits:
            logger.warning(
                f"Auto-submitting {len(self._batch_circuits)} remaining circuits"
            )
            self.submit_job()

        # Clean up batch context
        try:
            self._batch_context.__exit__(None, None, None)
        except Exception as e:
            logger.error(f"Error closing batch context: {e}")

        logger.info(
            f"Finished batch for {self._current_compiler.name} "
            f"({self._batch_job_count} jobs submitted)"
        )

        # Reset state
        self._batch_context = None
        self._current_compiler = None
        self._batch_circuits = []
        self._batch_job_count = 0

    def get_batch_stats(self) -> Dict:
        """Get statistics about current batch."""
        return {
            "active_batch": self._batch_context is not None,
            "current_compiler": self._current_compiler.name
            if self._current_compiler
            else None,
            "circuits_collected": len(self._batch_circuits),
            "jobs_submitted": self._batch_job_count,
        }

def build_grover_circuit(
    oracle: qiskit.QuantumCircuit, num_vars: int, grover_iterations: int
) -> qiskit.QuantumCircuit:
    """
    Build complete Grover search circuit from oracle.

    Args:
        oracle: Oracle circuit from problem_instance.oracle()
        num_vars: Number of search variables
        grover_iterations: Number of Grover iterations

    Returns:
        Complete Grover search circuit
    """
    # Build Grover operator
    grover_op = grover_operator(oracle, reflection_qubits=range(num_vars))

    # Create search circuit
    search_circuit = qiskit.QuantumCircuit(oracle.num_qubits, num_vars)

    # Initialize ancilla for Uf mode
    search_circuit.x(num_vars)
    search_circuit.h(num_vars)

    # Initialize superposition
    search_circuit.h(range(num_vars))

    # Apply Grover operator
    if grover_iterations > 0:
        search_circuit.compose(grover_op.power(grover_iterations), inplace=True)

    # Measure
    search_circuit.measure(range(num_vars), range(num_vars))

    return search_circuit  


def verify_oracle(oracle: qiskit.QuantumCircuit, problem: BaseProblem) -> bool:
    """
    Verify oracle correctness by checking its action on all basis states.

    Args:
        oracle: Oracle circuit for the problem in U_f oracle form
        num_vars: Number of input qubits (excluding any ancilla/result qubits)
    Returns:
        True if oracle behaves correctly, False otherwise
    """
    n = problem.number_of_input_bits()
    if n > 10:
        logger.warning(f"Attempting to verify a large oracle ({n} qubits). This could be resource-intensive.")
    
    simulator = AerSimulator()
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=simulator)
    oracle = pass_manager.run(oracle)

    for i in range(2**n):
        input_state = [(i >> j) & 1 for j in range(n)]
        qc = qiskit.QuantumCircuit(oracle.num_qubits, n+1)
        for q in range(n):
            if input_state[q]:
                qc.x(q)
        qc.compose(oracle, inplace=True)

        qc.measure(range(n+1), range(n+1))
        qc = pass_manager.run(qc)

        result = simulator.run(qc, shots=1024).result()
        counts = result.get_counts()

        expected_result = "1" if problem.verify_solution(input_state) else "0"
        expected_output = expected_result + f"{i:b}".zfill(n)

        # valid oracle if the majority counts are the expected output
        if counts.get(expected_output, 0) < 512:
            print(expected_output, counts)
            return False

    logger.info("Oracle verification passed")
    return True

def count_solutions(oracle: qiskit.QuantumCircuit, num_vars: int, backend: Optional[Backend] = None):
    """
    Estimates the number of solutions to f(x)=1 using Quantum Phase Estimation. 

    Args:
        oracle: Oracle circuit for the problem in U_f or phase oracle form
        n: Number of input qubits (excluding any ancilla/result qubits)
        backend: Backend for execution (if None, uses AerSimulator)

    Returns:
        Tuple of (estimated number of solutions f(x)=1, phase angle)

    """
    # Assume Uf mode (can change this to be a function argument to toggle for a phase oracle)
    uf_mode = True

    counting_qubits = num_vars
    counting_circuit = qiskit.QuantumCircuit(counting_qubits + oracle.num_qubits, counting_qubits)
    grover_op = grover_operator(oracle, reflection_qubits=range(num_vars))
    
    counting_circuit.h(range(counting_qubits))
    counting_circuit.h(range(counting_qubits, counting_qubits + num_vars))

    # initialize the result qubit to H |1> for phase oracle if uf_mode
    if uf_mode:
        counting_circuit.x(counting_qubits + num_vars)
        counting_circuit.h(counting_qubits + num_vars)
        
    for i in range(counting_qubits):
        power = 2**i
        controlled_grover = grover_op.power(power).control()
        counting_circuit.append(controlled_grover.to_instruction(),
                            [i] + list(range(counting_qubits, counting_qubits + oracle.num_qubits)))
    counting_circuit.append(QFT(counting_qubits, do_swaps=False).inverse(), range(counting_qubits))
    counting_circuit.measure(range(counting_qubits), range(counting_qubits))

    logger.info("Counting circuit constructed")

    if backend is None:
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=simulator)
        counting_circuit = pass_manager.run(counting_circuit)
        result = simulator.run(counting_circuit,shots=10**4).result()
        counts = result.get_counts()
    else:
        qc_transpiled = qiskit.transpile(counting_circuit, backend)
        sampler = Sampler(backend)
        logger.info(f"Submitting counting job to backend {backend.name()}")
        job = sampler.run([qc_transpiled], shots=10**4)
        result = job.result()[0]
        counts = result.data.c.get_counts()

    # extract the phase angle based on the most frequent counts measured
    output, count = max(counts.items(), key=lambda x: x[1])
    measured_value = int(output, 2)

    if measured_value > 2**(counting_qubits-1):
        # handle wrap-around
        measured_value = 2**counting_qubits - measured_value
    
    phase = measured_value / (2**counting_qubits)

    N = 2**num_vars
    m = N * (1 - math.cos(2 * math.pi * phase)) / 2
    return m, phase


# Utility functions for common operations
def calculate_grover_iterations(num_solutions: int, total_states: int) -> int:
    """
    Calculate optimal Grover iterations.

    For Grover's algorithm, the optimal number of iterations is:
    $$\\pi / (4 \\arcsin(\\sqrt{M/N}))$$

    where $M$ is the number of solutions and $N$ is the total number of states.

    This comes from the probability of success 
    $$P(k)=\\sin^2((2k+1)\theta)$$
    where k is the number of iterations,
    and $\theta = \\arcsin(\\sqrt{M/N})$

    Args:
        num_solutions: Number of solutions to the problem ($M$)
        total_states: Total number of possible states ($N = 2^n$)

    Returns:
        Optimal number of Grover iterations
    """
    if num_solutions == 0 or num_solutions == total_states:
        return 0

    ratio = num_solutions / total_states
    sqrt_ratio = math.sqrt(ratio)
    angle = math.asin(sqrt_ratio)
    result = math.pi / (4 * angle)

    def probability_of_success(k):
        return math.sin((2*k+1)*angle)**2
    
    p_floor = probability_of_success(math.floor(result))
    p_ceil = probability_of_success(math.ceil(result))

    print(
        f"Debug: M={num_solutions}, N={total_states}, ratio={ratio:.3f}, "
        f"angle={angle:.3f}, result={result:.3f}, floor={math.floor(result)} : {p_floor:.3f}, round={math.ceil(result)} : {p_ceil:.3f}"
    )

    if p_ceil > p_floor:
        return math.ceil(result)
    
    return math.floor(result)


