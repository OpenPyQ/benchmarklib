# Import the research library
import importlib.util
import itertools
import json
import logging
import math
import tempfile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit.providers import Backend
from sqlalchemy import JSON, String, select, func
from sqlalchemy.orm import Mapped, mapped_column, aliased
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction
from tweedledum.bool_function_compiler.decorators import circuit_input

from benchmarklib import BaseTrial, BaseProblem, BenchmarkDatabase, TrialCircuitMetricsMixin
from benchmarklib.algorithms.grover import build_grover_circuit, calculate_grover_iterations, verify_oracle
from benchmarklib.runners.queue import BatchQueue
from benchmarklib.runners.resource_management import run_with_resource_limits
from benchmarklib.core.types import classproperty
from benchmarklib.pipeline import PipelineCompiler
from benchmarklib.pipeline.config import PipelineConfig

logger = logging.getLogger("Clique.clique")

@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_cardinality(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    generate_at_least_k_counter(vertices, n, k)

    return s & at_least_k


@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_batcher(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size k in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    # generate_sorting_network(vertices, n, k)
    generate_batcher_sort_network(vertices, n, k)

    return s & sorted_bit_0


class SortPairNode:
    def __init__(self, high, low):
        self.high = high
        self.low = low

def get_sort_statements(variables):
    num_variables = len(variables)
    statements = []

    nodes = [[SortPairNode(None, None) for _ in range(num_variables)] for _ in range(num_variables)]
    for i in range(num_variables):
        nodes[i][0] = SortPairNode(variables[i], None)

    for i in range(1, num_variables):
        for j in range(1, i+1):
            s_high = f"s_{i}_{j}_high"
            s_low = f"s_{i}_{j}_low"
            nodes[i][j] = SortPairNode(s_high, s_low)

            if j == i:
                statements.append(f"{s_high} = {nodes[i-1][j-1].high} | {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j-1].high} & {nodes[i][j-1].high}")
            else:
                statements.append(f"{s_high} = {nodes[i-1][j].low} | {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j].low} & {nodes[i][j-1].high}")

    outputs = [nodes[num_variables-1][num_variables-1].high] + [nodes[num_variables-1][i].low for i in range(num_variables-1, 0, -1)]

    return statements, outputs

def construct_clique_verifier(graph, clique_size=None):
    """ 
    Given a graph in the form of binary string 
    e_11 e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_n-1n, returns the string of a python function that takes n boolean variables denoting vertices 
    True if in the clique and False if not,
    and returns whether the input is a clique of size at least n/2 in the graph.

    if clique_size is unspecified, the default is to require at least n/2 vertices
    """
    n = int((1 + (1 + 8*len(graph))**0.5) / 2)
    variables = [f'inpt[{i}]' for i in range(n)]
    statements, sort_outputs = get_sort_statements(variables)
    clique_size = clique_size or n//2

    # count whether there are at least clique_size vertices in the clique
    statements.append("count = " + sort_outputs[clique_size-1])

    # whenever there is not an edge between two vertices, they cannot both be in the clique
    if True:
        statements.append(f"edge_sat = {variables[0]} | ~ {variables[0]}") # this should be initialized to True, but qiskit classical function cannot yet parse True
    else:
        statements.append("edge_sat = True")
    edge_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge = graph[edge_idx]
            edge_idx += 1
            if edge == '0':
                # TODO: we could reduce depth to log instead of linear by applying AND more efficiently
                # for now, we'll let tweedledum optimize this
                statements.append(f"edge_sat = edge_sat & ~ ({variables[i]} & {variables[j]})")

    statements.append("return count & edge_sat")
    output = f"def verify(inpt: Tuple[bool]) -> bool:\n    "
    output += "\n    ".join(statements)
    return output

class CliqueProblem(BaseProblem):
    """
    Clique Problem Instance

    Args:
        g: Edge representation as binary string (e_12 e_13 ... e_1n e_23 ... e_(n-1)n)
        n: Number of vertices in the graph
        p: Edge probability (integer percentage, optional)
        clique_counts: Precomputed clique counts (optional, will compute if needed)
        instance_id: Database ID (None for unsaved instances)
    """
    __tablename__ = "clique_problems"
    TrialClass = "CliqueTrial"
    graph: Mapped[str] = mapped_column(String, unique=True)
    nodes: Mapped[int]
    edge_probability: Mapped[Optional[int]]
    _clique_counts: Mapped[Optional[List[int]]] = mapped_column(JSON)

    trials = None # disable relationship back-population as we have two kinds of trials

    def __init__(self, *args, **kwargs):
        if "clique_counts" in kwargs:
            kwargs["_clique_counts"] = kwargs.pop("clique_counts")
        super().__init__(*args, **kwargs)
        if not self._clique_counts:
            self.compute_clique_counts()

    @property
    def clique_counts(self) -> List[int]:
        """Get clique counts, computing if necessary."""
        if not self._clique_counts:
            self.compute_clique_counts()
        return self._clique_counts

    @property
    def target_clique_size(self) -> int:
        """Get target clique size (default n/2)."""
        return max(self.nodes // 2, 2)

    def compute_clique_counts(self) -> List[int]:
        """Compute the number of vertex subsets that form cliques of at least size k."""
        adjacency_matrix = self.as_adjacency_matrix()
        n = self.nodes
        clique_counts = [0 for _ in range(n + 1)]

        # All subsets are cliques of size 0
        clique_counts[0] = 2**n

        # All single vertices are cliques of size 1
        clique_counts[1] = n

        # Count edges for cliques of size 2
        clique_counts[2] = sum([1 for e in self.graph if e == "1"])

        # Count larger cliques
        for i in range(3, n + 1):
            for clique in itertools.combinations(range(n), i):
                if all(
                    adjacency_matrix[u, v] for u, v in itertools.combinations(clique, 2)
                ):
                    clique_counts[i] += 1

        # Make counts cumulative (at least k vertices in clique)
        for i in range(n - 1, 0, -1):
            clique_counts[i] += clique_counts[i + 1]

        self._clique_counts = clique_counts
        return clique_counts

    def as_adjacency_matrix(self) -> np.ndarray:
        """Convert edge representation to adjacency matrix."""
        adjacency_matrix = np.zeros((self.nodes, self.nodes))
        edge_idx = 0

        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "1":
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                edge_idx += 1

        return adjacency_matrix

    def verify_clique(self, vertex_assignment: str, clique_size: int) -> bool:
        """Verify if a vertex assignment represents a valid clique."""
        if len(vertex_assignment) != self.nodes:
            return False

        # Check if enough vertices are selected
        if sum(1 for v in vertex_assignment if v == "1") < clique_size:
            return False

        # Check that selected vertices form a clique
        edge_idx = 0
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "0":  # No edge between i and j
                    if vertex_assignment[i] == "1" and vertex_assignment[j] == "1":
                        return False  # Both selected but no edge
                edge_idx += 1

        return True

    def get_verifier_src(self) -> str:
        return construct_clique_verifier(self.graph, clique_size=self.target_clique_size)

    #### ProblemInstance Methods ####

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "graph": self.graph,
            "nodes": self.nodes,
            "edge_probability": self.edge_probability,
            "clique_counts": self._clique_counts,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], instance_id: Optional[int] = None
    ) -> "CliqueProblem":
        """Create instance from dictionary."""
        return cls(
            graph=data["graph"],
            nodes=data["nodes"],
            edge_probability=data.get("edge_probability"),
            clique_counts=data.get("clique_counts", []),
            instance_id=instance_id,
        )

    @classproperty
    def problem_type(cls) -> str:
        return "CLIQUE"

    def get_problem_size(self) -> Dict[str, int]:
        """Return key size metrics."""
        num_edges = sum(1 for e in self.graph if e == "1")
        return {
            "num_vertices": self.nodes,
            "num_edges": num_edges,
            "edge_probability": self.edge_probability or 0,
        }

    def number_of_input_bits(self) -> int:
        """Number of input bits for quantum oracle."""
        return self.nodes

    def get_number_of_solutions(self, trial: "CliqueTrial") -> int:
        clique_size = trial.clique_size
        if clique_size is None:
            raise ValueError(
                "No clique size for this trial, cannot compute number of solutions"
            )

        return self.clique_counts[clique_size]

class CliqueTrial(TrialCircuitMetricsMixin, BaseTrial):
    """Trial for clique detection using Grover's algorithm."""

    __tablename__ = "clique_trials"
    ProblemClass = CliqueProblem

    grover_iterations: Mapped[Optional[int]]
    clique_size: Mapped[int]

    def __init__(self, *args, **kwargs):
        # supply default clique_size from problem if not provided
        if "clique_size" not in kwargs and "problem" in kwargs:
            kwargs["clique_size"] = kwargs["problem"].target_clique_size
        super().__init__(*args, **kwargs)

    def calculate_expected_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate theoretical expected success rate."""

        grover_iterations = self.grover_iterations or 1
        clique_size = self.clique_size

        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Get number of solutions (cliques of at least the specified size)
        m = self.problem.clique_counts[clique_size]
        N = 2**self.problem.nodes

        if m == 0:
            return 0.0

        # Grover success probability calculation
        q = (2 * m) / N
        theta = math.atan(math.sqrt(q * (2 - q)) / (1 - q))
        phi = math.atan(math.sqrt(q / (2 - q)))

        return math.sin(grover_iterations * theta + phi) ** 2

    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate actual success rate from measurement results."""
        if self.is_failed:
            return 0.0
        
        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")

        clique_size = self.clique_size
        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Count successful measurements
        num_valid_cliques = 0
        total_shots = 0

        for measurement, count in self.counts.items():
            if measurement == "-1":  # Failed measurement
                total_shots += count
                continue

            # Reverse bit order to match graph representation
            reversed_measurement = measurement[::-1]

            if self.problem.verify_clique(reversed_measurement, clique_size):
                num_valid_cliques += count

            total_shots += count

        return num_valid_cliques / total_shots if total_shots > 0 else 0.0

#CliqueOracleProblem = aliased(CliqueProblem, name="clique_oracle_problems")

class CliqueOracleTrial(TrialCircuitMetricsMixin, BaseTrial):
    """Trial for measuring the output to a specific input to a clique oracle."""

    __tablename__ = "clique_oracle_trials"
    ProblemClass = CliqueProblem

    input_state: Mapped[str]
    expected_output: Mapped[Optional[bool]]

    def __init__(self, *args, **kwargs):
        if kwargs.get("is_failed", False):
            kwargs["input_state"] = "-1"
        super().__init__(*args, **kwargs)
        if self.expected_output is None:
            self.expected_output = self.problem.verify_clique(
                self.input_state, self.problem.target_clique_size
            )
        else:
            self.expected_output = self.expected_output

    def calculate_success_rate(self, *args, **kwargs) -> float:
        """Calculate success rate based on measurement results."""
        if self.is_failed:
            return 0.0

        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")
        
        total_shots = sum(self.counts.values())
        total_expected_output = (self.input_state + ("1" if self.expected_output else "0"))[::-1]  # reverse bit order for qiskit measurement
        successful_shots = self.counts.get(total_expected_output, 0)

        return successful_shots / total_shots if total_shots > 0 else 0.0


# Utility functions for creating problem instances


def create_random_graph_instance(
    n: int, p: int, compute_clique_counts: bool = True
) -> CliqueProblem:
    """Create a random graph instance."""
    import random

    num_edges = n * (n - 1) // 2
    g = "".join(["1" if random.random() * 100 < p else "0" for _ in range(num_edges)])

    instance = CliqueProblem(graph=g, nodes=n, edge_probability=p)

    if compute_clique_counts:
        instance.compute_clique_counts()

    return instance


def populate_clique_database(
    db: BenchmarkDatabase,
    n_range: range,
    p_range: List[int],
    graphs_per_config: int = 10,
) -> None:
    """Populate database with random clique problem instances."""
    for n in n_range:
        for p in p_range:
            for _ in range(graphs_per_config):
                instance = create_random_graph_instance(
                    n, p, compute_clique_counts=True
                )
                db.save_problem_instance(instance)


def _get_clique_trials(problem: CliqueProblem, compiler: "PipelineCompiler", config: PipelineConfig) -> Optional[List[CliqueTrial]]:
    """
    returns a list of CliqueTrial instances for the given problem and compiler
    """
    trials = []
    nodes = problem.nodes
    target_clique_size = max(nodes//2, 2)
    cliques_of_target_size = problem.clique_counts[target_clique_size]
    if cliques_of_target_size == 0:
        # clique of size target_clique_size DNE for this graph
        return []

    try:
        compile_result = compiler.compile(problem, clique_size=target_clique_size)
        if not compile_result.synthesis_circuit:
            raise Exception("No synthesis circuit returned")
    except Exception as e:
        logger.error(f"Compilation failed for problem ID {problem.id} with error: {e}")
        return None
    oracle = compile_result.synthesis_circuit

    # verify small oracles as a sanity check (but skip large ones which take too long in simulation)
    if problem.nodes <= 5 and not verify_oracle(oracle, problem):
        logger.warning(f"Oracle verification failed for problem ID {problem.id}, skipping trial.")
        return None

    optimal_grover_iters = calculate_grover_iterations(cliques_of_target_size, 2**nodes)
    for grover_iters in range(1, optimal_grover_iters):

        circuit = build_grover_circuit(oracle, problem.number_of_input_bits(), grover_iters)
        circuit_transpiled = compiler.transpile(circuit)

        trial = CliqueTrial(
            problem=problem,
            compiler_name=compiler.name,
            grover_iterations=grover_iters,
            clique_size=target_clique_size,
            pipeline_config=config,
            circuit=circuit_transpiled,
            circuit_pretranspile=circuit,
        )
        trials.append(trial)

    return trials

def _get_clique_oracle_trials(problem: CliqueProblem, compiler: "PipelineCompiler", config: PipelineConfig, num_trials: int = 5) -> Optional[List[CliqueOracleTrial]]:
    """
    returns a list of CliqueOracleTrial instances for the given problem and compiler
    """
    trials = []
    nodes = problem.nodes
    target_clique_size = problem.target_clique_size

    try:
        compile_result = compiler.compile(problem, clique_size=target_clique_size)
        if not compile_result.synthesis_circuit:
            raise Exception("No synthesis circuit returned")
    except Exception as e:
        logger.error(f"Compilation failed for problem ID {problem.id} with error: {e}")
        return None
    oracle = compile_result.synthesis_circuit

    # verify small oracles as a sanity check (but skip large ones which take too long in simulation)
    #if problem.nodes <= 5 and not verify_oracle(oracle, problem):
    #    logger.warning(f"Oracle verification failed for problem ID {problem.id}, skipping trial.")
    #    return None

    import random
    for _ in range(num_trials):
        input_state = ''.join(random.choice(['0', '1']) for _ in range(nodes))
        qc = qiskit.QuantumCircuit(max(oracle.num_qubits, problem.nodes + 1), problem.nodes + 1)
        for i, bit in enumerate(input_state):
            if bit == '1':
                qc.x(qc.qubits[i])

        qc.compose(oracle, inplace=True)
        qc.measure(range(problem.nodes + 1), range(problem.nodes + 1))

        transpiled_qc = compiler.transpile(qc)
        trials.append(CliqueOracleTrial(
                problem=problem,
                compiler_name=compiler.name,
                input_state=input_state,
                pipeline_config=config,
                circuit = transpiled_qc,
                circuit_pretranspile = qc,
            )
        )

    return trials

def run_clique_benchmark(db: BenchmarkDatabase, compiler: "PipelineCompiler", backend: Backend, nodes_iter: Iterable[int], num_problems: int = 20, shots: int = 10**3, max_problems_per_job: Optional[int] = None, save_circuits: bool = False):
    # get compiler pipeline config to save with each trial
    config = db.get_saved_config(compiler.config)

    assert backend.name == config.backend.name

    with BatchQueue(db, backend=backend, shots=shots) as q:
        for nodes in nodes_iter:
                
            count = db.query(
                select(func.count(func.distinct(db.problem_class.id)))
                .select_from(db.problem_class)
                        .join(db.trial_class, db.trial_class.problem_id == db.problem_class.id)
                        .where(
                            db.trial_class.pipeline_config == config,
                            db.problem_class.nodes == nodes,
                    )
                )[0]
            if count >= num_problems:
                logger.info(f"Skipping (nodes={nodes}) -- already have {count} instances")
                continue

            pending_trial_problem_count = 0

            for problem in db.find_problem_instances(
                nodes=nodes,
                limit=num_problems - count, 
                compiler_name=compiler.name,
                choose_untested=True,
                random_sample=True
            ):
                logger.info(f"Compiling problem ID {problem.id} with {nodes} nodes.")
                compiler_run = run_with_resource_limits(
                    _get_clique_trials if db.trial_class == CliqueTrial else _get_clique_oracle_trials,
                    kwargs={
                        "problem": problem,
                        "compiler": compiler,
                        "config": config,
                    },
                    memory_limit_mb=2024,
                    timeout_seconds=240
                )
                trials = compiler_run.result if compiler_run.success else None
                if trials is None:
                    logger.warning(f"Compilation failed for problem ID {problem.id}: {compiler_run.error_message}.")
                    db.create_compilation_failure(problem, compiler.name)
                    continue
                
                for trial in trials:
                        qc = trial.circuit_pretranspile
                        transpiled_qc = trial.circuit
                        if not save_circuits:
                            trial.circuit = None
                            trial.circuit_pretranspile = None
                        q.enqueue(trial, transpiled_qc, run_simulation=(qc.num_qubits <= 10))

                pending_trial_problem_count += 1
                if max_problems_per_job and pending_trial_problem_count >= max_problems_per_job:
                    q.submit_tasks()
                    pending_trial_problem_count = 0


def run_clique_benchmark_sample(db: BenchmarkDatabase, compiler: "PipelineCompiler", backend: Backend, problems: Iterable[CliqueProblem], shots: int = 10**3, max_problems_per_job: Optional[int] = None, save_circuits: bool = False):
    """
    Alternative version of run_clique_benchmark that takes a list of problems directly instead of finding untested problems with the specified range of nodes (run_clique_benchmark).
    """
    # get compiler pipeline config to save with each trial
    config = db.get_saved_config(compiler.config)
    assert backend.name == config.backend.name


    pending_trial_problem_count = 0
    with BatchQueue(db, backend=backend, shots=shots) as q:
        for problem in problems:
            compiler_run = run_with_resource_limits(
                _get_clique_trials if db.trial_class == CliqueTrial else _get_clique_oracle_trials,
                kwargs={
                    "problem": problem,
                    "compiler": compiler,
                    "config": config,
                },
                memory_limit_mb=2024,
                timeout_seconds=240
            )
            trials = compiler_run.result if compiler_run.success else None
            if trials is None:
                logger.warning(f"Compilation failed for problem ID {problem.id}: {compiler_run.error_message}.")
                db.create_compilation_failure(problem, compiler.name)
                continue
            
            for trial in trials:
                    qc = trial.circuit_pretranspile
                    transpiled_qc = trial.circuit
                    if not save_circuits:
                        trial.circuit = None
                        trial.circuit_pretranspile = None
                    q.enqueue(trial, transpiled_qc, run_simulation=(qc.num_qubits <= 10))

            pending_trial_problem_count += 1
            if max_problems_per_job and pending_trial_problem_count >= max_problems_per_job:
                q.submit_tasks()
                pending_trial_problem_count = 0
    

