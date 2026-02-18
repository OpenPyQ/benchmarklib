from __future__ import annotations  # needed for type hinting without circular imports
import ast

# implementation of
import logging
import sys
import tempfile
from typing import Optional

import networkx as nx
import pygraphviz
from qiskit import QuantumCircuit

from benchmarklib.pipeline.synthesis import Synthesizer
from benchmarklib.pipeline.registries import SynthesizerRegistry

### if qcompiler module installed
try:
    import qcompiler
except ImportError:
    raise ImportError(
        "QuantumCompilerMPC Not installed"
        "Install QuantumCompilerMPC https://github.com/whyster/QuantumCompilerMPC"
    ) from None

from pathlib import Path

# QuantumCompilerMPC/compiler/compiler'
qcompiler_dir = Path(qcompiler.__path__[0])
# WARN: Super janky, we need to have the compiler accept inputs so we can avoid this
QMPC_BASE_DIR = qcompiler_dir.parent.parent
CLIQUE_VERIFIER_CONSTANTS = str(
    QMPC_BASE_DIR / "quantum_benchmarks/clique_verifier.constants"
)
CLIQUE_VERIFIER_PY = str(QMPC_BASE_DIR / "quantum_benchmarks/clique_verifier.py")

logger = logging.getLogger("benchmarklib.pipeline.synthesis.qmpc")

try:
    from tweedledum import converters
    from tweedledum.bool_function_compiler.decorators import circuit_input
    from tweedledum.bool_function_compiler.quantum_circuit_function import (
        QuantumCircuitFunction,
    )
    from tweedledum import BitVec
    from tweedledum.classical import optimize
    from tweedledum.passes import linear_resynth, parity_decomp
    from tweedledum.synthesis import xag_cleanup, xag_synth
    from tweedledum.utils import xag_export_dot
except ImportError:
    logger.warning("Tweedledum not installed, QuantumMPC synthesis will not work.")


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger):
        self.logger = logger
        self.linebuf = ""

    def write(self, buf):
        """
        Receives chunks of output. Splits them into lines and logs each line.
        """
        for line in buf.rstrip().splitlines():
            # Check if line is not empty (e.g., from an empty print() call)
            if line:
                self.logger.debug(line.rstrip())

    def flush(self):
        # This method is often called by internal Python I/O, but we don't
        # need to do anything specific here since we log line by line in write().
        pass


@SynthesizerRegistry.register
class QuantumMPC(Synthesizer):
    """
    Synthesis using QuantumMPC Compiler

    This compiler:
    1. Creates a python function using QuantumMPC Compiler
    2. Converts to XAG representation
    3. Synthesizes using xag_synth
    4. Applies optimization passes
    5. Converts to phase-flip oracle
    """

    def __init__(self):
        """
        Initialize compiler.
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
        from benchmarklib.problems import CliqueProblem
        # Determine which classical function to use based on problem type
        if isinstance(problem, CliqueProblem) or problem.problem_type.lower() == "clique":
            return self._compile_clique(problem, **kwargs)
        else:
            return self.compile_src(src=problem.get_verifier_src(), n=problem.number_of_input_bits(), params=problem.get_verifier_params())
        
    def compile_from_file(self, file_path: str, n: int, function_name: Optional[str]=None, src: Optional[str]=None, params: Optional[dict]=None):
        """
        Compile a function from a source code file using QuantumMPC compiler.

        :param file_path: Path to file containing function to compile.
        :type file_path: str
        :param n: Number of input bits to the function
        :type n: int
        :param function_name: Name of the function to compile. Note: compiler currently only supports one function per file.
        :type function_name: Optional[str]
        :param params: Compile time constants to provide as input to the function
        :type params: Optional[dict]
        :param src: Optional source code string to compile. If None, will read from file_path. Provide this to avoid reading file multiple times.
        :type src: Optional[str]

        :return: QuantumCircuit representing the compiled function as a Uf oracle
        :rtype: qiskit.QuantumCircuit
        """
        
        if src is None:
            with open(file_path, "r") as f:
                src = f.read()

        if function_name is None:
            # deduce function name from src
            func_defs = [line for line in src.split("\n") if line.strip().startswith("def ")]
            if len(func_defs) != 1:
                raise ValueError("function_name must be specified if multiple functions are present in the source file")
            function_name = func_defs[0].split("def ")[1].split("(")[0].strip()

        
        linear = str(qcompiler.compile(
            file_path,
            src,
            quiet=True,
            run_vectorization=False,
            backend=None,
            constants=params,
        ))


        # TODO: the following code is such a hack
        # it needs a complete rewrite when we get a better
        # API from the compiler and from tweedledum / lower-level synthesis

        # deduce quantum input variable name
        # Note: we are only considering single tuple/list of bools as input for now
        try:
            input_var_name = linear.split(": shared")[0].split("def ")[1].split("(")[1].strip()
            input_var_name = input_var_name.replace("!", "_ex_")  # fix invalid naming scheme from compiler output
        except Exception as e:
            print(e)
            raise Exception("Failed to deduce name of the input variable for the oracle.")
        

        # adjust linear to ensure the first variable used is also the result variable
        # so that Tweedledum synthesizes the oracle to correctly have the result qubit as the n+1 qubit
        linear_lines = str(linear).split("\n")
        indent = linear_lines[1][: linear_lines[1].index(linear_lines[1].lstrip())]
        new_return_val = "_result_result_mpc"
        linear_lines.insert(1, f"{indent}{new_return_val} = True")
        original_return_val = linear_lines[-1].split("return ")[1].strip()
        linear_lines[-1] = f"{indent}{new_return_val} = {new_return_val} and {original_return_val}"
        linear_lines.append(f"{indent}return {new_return_val}")
        linear = "\n".join(linear_lines)

        # Take linear, and remove the function definition and replace it with our own
        func_def = f"@circuit_input({input_var_name}=lambda n: BitVec(n))\ndef verifier_qmpc(n={n}) -> BitVec(1):"
        classical_func_def = (
            f"def classical_verifier_qmpc({input_var_name}:list[bool]) -> bool:"
        )

        qc_verifier_source = self.string_to_python_code_obj(
            linear, f"verifier_qmpc_compiled", func_def
        )
        _ = self.string_to_python_code_obj(
            linear, f"classical_verifier_qmpc_compiled", classical_func_def
        )

        # verifier_qmpc is a function in our python environment, as executed by string_to_python_code_obj,
        # NOTE: each time this is run, the previous verifier_qmpc is overwritten
        verifier_qmpc.__source__ = qc_verifier_source
        
        qc_func = QuantumCircuitFunction(verifier_qmpc)
        self.compilation_artifacts["source"] = qc_func.get_transformed_source()
        
        self.compilation_artifacts["classical_verifier"] = (
            classical_verifier_qmpc
        )

        # Get XAG and optionally optimize
        xag = qc_func.logic_network()

        logger.debug("Optimizing XAG...")
        xag = xag_cleanup(xag)
        optimize(xag)

        with tempfile.NamedTemporaryFile() as fp:
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

        qiskit_circuit = converters.to_qiskit(td_circuit, circuit_type="gatelist")

        return qiskit_circuit

    def compile_src(self, src: str, n: int, params: Optional[dict]=None):
        """
        Compile a function from a source code string using QuantumMPC compiler.

        :param src: Source code containing the function to compile
        :type src: str
        :param n: Number of input bits to the function
        :type n: int
        :param params: Compile time constants to provide as input to the function
        :type params: Optional[dict]

        :return: QuantumCircuit representing the compiled function as a Uf oracle
        :rtype: qiskit.QuantumCircuit
        """
        
        # update function header to match QMPC expectations
        # note that shared does not have to be imported, just needs to be present in the type annotations
        src = src.replace("Tuple[bool]", "shared[list[bool]]")
        src = src.replace("-> bool", "-> shared[bool]")
        
        # Write function to temporary file and compile from there
        with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
            temp_file.write(src.encode())
            temp_file.flush()
            return self.compile_from_file(temp_file.name, n=n, src=src, params=params)

    def string_to_python_code_obj(
        self, source: str, function_name: str, function_header: str
    ):
        """
        Convert compiler output string to executable python code and execute it to place the function within the interpreter environment.
        """
        # Take linear, and remove the function definition and replace it with our own
        new_source = str(source).split("\n")[1:]
        new_source.insert(0, function_header)

        new_source = "\n".join(new_source)
        new_source = new_source.replace("!", "_ex_")

        func_ast = ast.parse(new_source)

        code_obj = compile(func_ast, function_name, mode="exec")

        exec(code_obj, globals())
        return new_source

    def _compile_clique(self, problem: CliqueProblem, **kwargs) -> QuantumCircuit:
        """Compile clique problem to oracle."""

        def convert_problem_to_input_file(p: CliqueProblem, k: int):
            adj = p.as_adjacency_matrix()
            N = adj.shape[0]
            edges = list(map(lambda x: bool(x), adj.flatten()))
            input = f"N!0={N}\nK!0={k}\nEdges!0={edges}"
            return input

        # redirect compiler stdout to logger to silence it
        original_stdout = sys.stdout
        sys.stdout = StreamToLogger(logger)

        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified for clique problems")

        # Create QuantumCircuitFunction
        n = problem.nodes

        # 'w' mode erases current contents
        with open(CLIQUE_VERIFIER_CONSTANTS, "w") as input_f:
            input_file_contents = convert_problem_to_input_file(
                problem, clique_size
            )
            print("INPUT FILE CONTENTS: ", input_file_contents)
            input_f.write(input_file_contents)

        with open(CLIQUE_VERIFIER_PY, "r") as f:
            src = f.read()

        linear = qcompiler.compile(
            str(CLIQUE_VERIFIER_PY),
            # str(__CLIQUE_VERIFIER_CONSTANTS),
            src,
            None,
            False,
            False,  # Do not output vectorized code. If True, only outputs vectorized code when invoked from the command line
        )

        # adjust linear to ensure the first variable used is also the result variable
        # so that Tweedledum synthesizes the oracle to correctly have the result qubit as the n+1 qubit
        linear_lines = str(linear).split("\n")
        indent = linear_lines[1][: linear_lines[1].index(linear_lines[1].lstrip())]
        new_return_val = "_result_result_mpc"
        linear_lines.insert(1, f"{indent}{new_return_val} = True")
        original_return_val = linear_lines[-1].split("return ")[1].strip()
        linear_lines[-1] = f"{indent}{new_return_val} = {new_return_val} and {original_return_val}"
        linear_lines.append(f"{indent}return {new_return_val}")
        linear = "\n".join(linear_lines)

        # Take linear, and remove the function definition and replace it with our own
        func_def = f"@circuit_input(Vertices_ex_0=lambda n: BitVec(n))\ndef clique_verifier_qmpc(n={n}) -> BitVec(1):"
        classical_func_def = (
            f"def clique_verifier_qmpc_classical(Vertices_ex_0:list[bool]) -> bool:"
        )

        qc_verifier_source = self.string_to_python_code_obj(
            linear, "clique_verifier_compiled", func_def
        )
        _ = self.string_to_python_code_obj(
            linear, "classical_clique_verifier_compiled", classical_func_def
        )

        # clique_verifier_qmpc is a function in our python environment,
        # NOTE: each time this is run, the previous clique_verifier_qmpc is overwritten
        clique_verifier_qmpc.__source__ = qc_verifier_source
        print(qc_verifier_source)
        qc_func = QuantumCircuitFunction(clique_verifier_qmpc)
        self.compilation_artifacts["source"] = qc_func.get_transformed_source()
        print(qc_func.get_transformed_source())
        self.compilation_artifacts["classical_verifier"] = (
            clique_verifier_qmpc_classical
        )

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
        #td_circuit = linear_resynth(td_circuit)  # warning: linear_resynth occasionally produces invalid results (non-equivalent circuit)

        # Convert to Qiskit
        qiskit_circuit = converters.to_qiskit(td_circuit, circuit_type="gatelist")

        # Convert to phase-flip oracle
        # The oracle qubit is the last qubit (output of the function)
        self.oracle_qubit = n

        #phase_oracle = QuantumCircuit(qiskit_circuit.num_qubits)
        #phase_oracle.x(self.oracle_qubit)
        #phase_oracle.h(self.oracle_qubit)
        #phase_oracle.compose(qiskit_circuit, inplace=True)
        #phase_oracle.h(self.oracle_qubit)
        #phase_oracle.x(self.oracle_qubit)

        # restore stdout
        sys.stdout = original_stdout

        return qiskit_circuit#phase_oracle

    def target_qubit(self) -> Optional[int]:
        """
        Return the index of the target qubit of the oracle
        """

        return self.oracle_qubit if self.oracle_qubit else None
