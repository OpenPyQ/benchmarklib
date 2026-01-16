from __future__ import annotations  # needed for type hinting without circular imports

import logging
import tempfile
import pygraphviz
import networkx as nx
from typing import Optional

from qiskit import QuantumCircuit

from .synthesizer import Synthesizer
from ..registries import SynthesizerRegistry

logger = logging.getLogger("benchmarklib.pipeline.synthesis.xag")

try:
    import tweedledum as td
    from tweedledum.bool_function_compiler import QuantumCircuitFunction
    from tweedledum.classical import optimize
    from tweedledum.passes import linear_resynth, parity_decomp
    from tweedledum.synthesis import xag_cleanup, xag_synth
    from tweedledum.utils import xag_export_dot

    from .clique_oracle import clique_oracle
except ImportError:
    logger.warning("Tweedledum not installed, XAG synthesis will not work.")


@SynthesizerRegistry.register
class XAGSynthesizer(Synthesizer):
    """
    Synthesis using XAG (XOR-AND Graph) synthesis from Tweedledum.

    This compiler:
    1. Creates a QuantumCircuitFunction from the problem's classical function
    2. Optionally optimizes the XAG representation
    3. Synthesizes using xag_synth
    4. Applies optimization passes
    5. Converts to phase-flip oracle
    """

    def __init__(self):
        """
        Initialize XAG compiler.
        """
        self.compilation_artifacts = {}

    @property
    def name(self) -> str:
        return str(self.__class__.__name__)

    def synthesize(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
        """
        Compile problem instance to phase-flip oracle using XAG synthesis.

        Args:
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters (e.g., clique_size)

        Returns:
            Phase-flip oracle quantum circuit
        """
        # Determine which classical function to use based on problem type
        from ...problems import CliqueProblem
        if isinstance(problem, CliqueProblem) or problem.problem_type == "CLIQUE":
            return self._compile_clique(problem, **kwargs)
        else:
            raise NotImplementedError(
                f"XAGCompiler doesn't support {problem.problem_type} problems yet"
            )

    def _compile_clique(self, problem: CliqueProblem, **kwargs) -> QuantumCircuit:
        """Compile clique problem to oracle."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified for clique problems")

        param_func = clique_oracle

        # Get edge list from problem
        edges = problem.as_adjacency_matrix().flatten().tolist()

        # Create QuantumCircuitFunction
        n = problem.nodes
        classical_inputs = {"n": n, "k": clique_size, "edges": edges}
        qc_func = QuantumCircuitFunction(param_func, **classical_inputs)
        self.compilation_artifacts["source"] = qc_func.get_transformed_source()

        # Get XAG and optionally optimize
        xag = qc_func.logic_network()

        logger.debug("Optimizing XAG...")
        xag = xag_cleanup(xag)
        optimize(xag)

        fp = tempfile.NamedTemporaryFile()
        logger.debug(f"Outputting XAG to tempfile {fp.name}")
        xag_export_dot(xag, fp.name)
        graphviz_str = fp.read()
        logger.debug(graphviz_str)

        agraph = pygraphviz.AGraph().from_string(graphviz_str)

        self.compilation_artifacts["nxag"] = nx.nx_agraph.from_agraph(agraph)

        # Synthesize
        logger.debug("Synthesizing from XAG...")
        td_circuit = xag_synth(xag)

        # Apply Tweedledum optimization passes
        logger.debug("Applying optimization passes...")
        td_circuit = parity_decomp(td_circuit)
        td_circuit = linear_resynth(td_circuit)

        # Convert to Qiskit
        qiskit_circuit = td.converters.to_qiskit(td_circuit, circuit_type="gatelist")

        # Convert to phase-flip oracle
        # The oracle qubit is the last qubit (output of the function)
        self.oracle_qubit = n

        phase_oracle = QuantumCircuit(qiskit_circuit.num_qubits)
        phase_oracle.x(self.oracle_qubit)
        phase_oracle.h(self.oracle_qubit)
        phase_oracle.compose(qiskit_circuit, inplace=True)
        phase_oracle.h(self.oracle_qubit)
        phase_oracle.x(self.oracle_qubit)

        return phase_oracle

    def target_qubit(self) -> Optional[int]:
        """
        Return the index of the target qubit of the oracle
        """

        return self.oracle_qubit if self.oracle_qubit else None
