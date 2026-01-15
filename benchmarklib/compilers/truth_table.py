import logging
from typing import Optional
import tempfile
import importlib.util
import os

import tweedledum as td
from qiskit import QuantumCircuit
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input
from tweedledum.passes import linear_resynth, parity_decomp
from tweedledum.synthesis import pkrm_synth
from tweedledum import BitVec

from .base import SynthesisCompiler, clique_oracle
from ..core import BaseProblem
from ..problems import CliqueProblem

logger = logging.getLogger("benchmarklib.compiler.truth_table")


class TruthTableCompiler(SynthesisCompiler):
    """
    Compiler using truth table synthesis (PKRM - Positive-polarity Reed-Muller).

    This approach:
    1. Simulates the classical function to get its truth table
    2. Uses PKRM synthesis for the truth table
    3. Applies optimization passes
    4. Converts to phase-flip oracle

    Note: Truth table approach scales exponentially with problem size!
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def compile(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
        """
        Compile problem instance to phase-flip oracle using truth table synthesis.

        Args:
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters

        Returns:
            Phase-flip oracle quantum circuit
        """
        if isinstance(problem, CliqueProblem):
            # support Clique problems directly
            return self._compile_clique(problem, **kwargs)
        
        # for other problems, try to use the verifier function
        logger.debug("Compiling the problem verifier source")
        src = problem.get_verifier_src()
        
        src = src.replace("def verify(inpt: Tuple[bool]) -> bool:", "def verify() -> BitVec(1):")
        with tempfile.TemporaryDirectory() as temp_dir:
            module_name = "temp_boolean_func"
            file_path = os.path.join(temp_dir, f"{module_name}.py")

            with open(file_path, "w") as f:
                f.write("from typing import *\n")
                f.write("from tweedledum import BitVec\n")
                f.write(src)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)
            verifier = QuantumCircuitFunction(circuit_input(inpt=BitVec(problem.number_of_input_bits()))(temp_module.verify))
            return verifier.truth_table_synthesis()


    def _compile_clique(self, problem: CliqueProblem, **kwargs) -> QuantumCircuit:
        """Compile clique problem using truth table synthesis."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified for clique problems")

        # Choose the parameterized function
        param_func = clique_oracle

        # Get edge list from problem
        edges = problem.as_adjacency_matrix().flatten().tolist()

        # Create QuantumCircuitFunction
        n = problem.nodes
        classical_inputs = {"n": n, "k": clique_size, "edges": edges}
        qc_func = QuantumCircuitFunction(param_func, **classical_inputs)

        # Simulate to get truth table
        logger.debug("Simulating to get truth table...")
        qc_func.simulate_all()

        # Synthesize from truth table
        logger.debug("Synthesizing from truth table...")
        td_circuit = pkrm_synth(qc_func._truth_table[0])

        # Apply optimization passes
        logger.debug("Applying optimization passes...")
        td_circuit = parity_decomp(td_circuit)
        td_circuit = linear_resynth(td_circuit)

        # Convert to Qiskit
        qiskit_circuit = td.converters.to_qiskit(td_circuit, circuit_type="gatelist")

        # Convert to phase-flip oracle
        oracle_qubit = qiskit_circuit.num_qubits - 1

        phase_oracle = QuantumCircuit(qiskit_circuit.num_qubits)
        phase_oracle.x(oracle_qubit)
        phase_oracle.h(oracle_qubit)
        phase_oracle.compose(qiskit_circuit, inplace=True)
        phase_oracle.h(oracle_qubit)
        phase_oracle.x(oracle_qubit)

        return phase_oracle

    def target_qubit(self) -> Optional[int]:
        """
        Return the index of the target qubit of the oracle
        """

        return self.oracle_qubit if self.oracle_qubit else None

class QCFCompiler(TruthTableCompiler):
    """
    Alias for TruthTableCompiler for consistency with database compiler_name.
    Qiskit Classical Function (QCF) Compiler using truth table synthesis.
    """

    @property
    def name(self) -> str:
        return "CLASSICAL_FUNCTION"