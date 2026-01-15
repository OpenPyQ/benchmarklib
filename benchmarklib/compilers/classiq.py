import logging
from math import ceil, log2

import networkx as nx
from classiq import (
    Constraints,
    Output,
    Preferences,
    QArray,
    QNum,
    allocate,
    create_model,
    qfunc,
    synthesize,
)
from qiskit import QuantumCircuit

from .base import SynthesisCompiler
from ..core import BaseProblem
from ..problems import CliqueProblem

logger = logging.getLogger("benchmarklib.compiler.classiq")


class ClassiqCompiler(SynthesisCompiler):
    """
    Compiler using Classiq Synthesis.

    This compiler:
    1. Creates a QMod instance from the problem's classical function
    4. Applies optimization passes
    5. Converts to phase-flip oracle
    """

    def __init__(self):
        """
        Initialize Classiq compiler.
        """

    @property
    def name(self) -> str:
        return "CLASSIQ"

    def compile(self, problem: BaseProblem, **kwargs):
        """
        Compile problem instance to phase-flip oracle using XAG synthesis.

        Args:
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters (e.g., clique_size)

        Returns:
            Phase-flip oracle quantum circuit
        """
        # Determine which classical function to use based on problem type
        if isinstance(problem, CliqueProblem):
            return self._compile_clique(problem, **kwargs)
        else:
            raise NotImplementedError(
                f"XAGCompiler doesn't support {problem.problem_type} problems yet"
            )

    def _compile_clique(self, problem: CliqueProblem, **kwargs):
        """Compile clique problem to oracle."""

        # extract info from problem and create graph
        G = nx.from_numpy_array(problem.as_adjacency_matrix())
        N = G.number_of_nodes()

        k = kwargs.get("clique_size")
        if k is None:
            raise ValueError("clique_size must be specified for clique problems")

        @qfunc
        def main(vertices: Output[QArray[QNum[1]]], oracle_result: Output[QNum[1]]):
            """
            Main function for Grover's algorithm setup.
            """

            # Allocate and initialize vertices
            allocate(N, vertices)

            # Allocate oracle result qubit
            cond = 1

            for i in range(N):
                for j in range(i + 1, N):
                    if not G.has_edge(i, j):
                        cond &= ~(vertices[i] & vertices[j])
            print(cond)

            # Properly allocate and initialize counter
            COUNTER_SIZE = ceil(log2(N + 1))
            # Need to count from 0 to N
            counter = QNum("counter", COUNTER_SIZE)  # Allocate with proper size
            allocate(COUNTER_SIZE, counter)

            for i in range(vertices.len):
                counter += vertices[i]

            cond_count = counter >= k
            cond &= cond_count

            oracle_result |= cond

            for i in range(vertices.len):
                counter += -vertices[i]

            # syntheis

        qmod = create_model(main)

        constraints = Constraints(optimization_parameter="depth")

        prefs = Preferences(
            optimization_level=3,
        )

        # qprog = synthesize(qmod, preferences=prefs, constraints=constraints)
        qprog = synthesize(qmod)
        oracle = QuantumCircuit.from_qasm_str(qprog.qasm)
        phase_oracle = QuantumCircuit(oracle.num_qubits)

        # Convert to phase-flip oracle
        # The oracle qubit is the last qubit (output of the function)
        oracle_qubit = oracle.num_qubits - 7

        phase_oracle.x(oracle_qubit)
        phase_oracle.h(oracle_qubit)
        phase_oracle.compose(oracle, inplace=True)
        phase_oracle.h(oracle_qubit)
        phase_oracle.x(oracle_qubit)

        return phase_oracle
