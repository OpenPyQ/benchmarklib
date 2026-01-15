# Import the research library
import itertools
import json
import math
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from ..core import BaseTrial, BenchmarkDatabase, BaseProblem
from ..core.types import _ProblemInstance, _BaseTrial


class CliqueProblem(_ProblemInstance):
    def __init__(
        self,
        graph: str,
        nodes: int,
        edge_probability: Optional[int] = None,
        clique_counts: Optional[List[int]] = None,
        instance_id: Optional[int] = None,
    ):
        """
        Clique Problem Instance

        Args:
            g: Edge representation as binary string (e_12 e_13 ... e_1n e_23 ... e_(n-1)n)
            n: Number of vertices in the graph
            p: Edge probability (integer percentage, optional)
            clique_counts: Precomputed clique counts (optional, will compute if needed)
            instance_id: Database ID (None for unsaved instances)
        """
        super().__init__(instance_id)

        self.graph = graph
        self.nodes = nodes
        self.edge_probability = edge_probability
        self._clique_counts = clique_counts or []

        # Validate edge representation
        expected_edges = nodes * (nodes - 1) // 2
        if len(graph) != expected_edges:
            raise ValueError(
                f"Invalid edge representation: expected {expected_edges} edges, got {len(graph)}"
            )

    @property
    def clique_counts(self) -> List[int]:
        """Get clique counts, computing if necessary."""
        if not self._clique_counts:
            self.compute_clique_counts()
        return self._clique_counts

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

    def from_nx_graph(
        graph: nx.Graph,
        edge_probability: Optional[int],
        clique_counts: Optional[List[int]],
    ):
        """Convert networkx graph into CliqueProblem for use with this library"""
        graph_str = ""
        nodes = list(graph.nodes())
        for i, _ in enumerate(nodes):
            for j in nodes[i + 1 :]:
                graph_str += "1" if graph.has_edge(i, j) else "0"

        return CliqueProblem(
            graph=graph_str,
            nodes=len(nodes),
            edge_probability=edge_probability,
            clique_counts=clique_counts,
            instance_id=None,
        )

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

    # ProblemInstance Methods ####

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

    @property
    def problem_type(self) -> str:
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

    def get_number_of_solutions(self, **trial_params) -> int:
        clique_size = trial_params.get("clique_size", None)
        if clique_size is None:
            raise ValueError(
                "No clique size for this trial, cannot compute number of solutions"
            )

        return self.clique_counts[clique_size]


class CliqueTrial(_BaseTrial):
    """Trial for clique detection using Grover's algorithm."""

    def calculate_expected_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate theoretical expected success rate."""
        if self._problem_instance is None:
            if db_manager is None:
                raise ValueError(
                    "Either problem_instance or db_manager must be provided"
                )
            self._problem_instance = self.get_problem_instance(db_manager)

        grover_iterations = self.trial_params.get("grover_iterations", 1)
        clique_size = self.trial_params.get("clique_size")

        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Get number of solutions (cliques of at least the specified size)
        m = self._problem_instance.clique_counts[clique_size]
        N = 2**self._problem_instance.nodes

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
        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")

        if self.is_failed:
            return 0.0

        # Load problem instance if needed
        if self._problem_instance is None:
            if db_manager is None:
                raise ValueError(
                    "Either problem_instance or db_manager must be provided"
                )
            self._problem_instance = self.get_problem_instance(db_manager)

        clique_size = self.trial_params.get("clique_size")
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

            if self._problem_instance.verify_clique(reversed_measurement, clique_size):
                num_valid_cliques += count

            total_shots += count

        return num_valid_cliques / total_shots if total_shots > 0 else 0.0


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


# Example usage and migration helpers


def migrate_old_graph_to_new_system(old_graph_db_path: str, new_db_path: str) -> None:
    """Migrate graphs from old system to new unified system."""
    import sqlite3

    # Create new database
    new_db = BenchmarkDatabase(new_db_path, CliqueProblem, CliqueTrial)

    count = 0
    # Read from old database
    with sqlite3.connect(old_graph_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT g, p, n, clique_counts FROM graphs")

        for row in cursor.fetchall():
            g, p, n, clique_counts_json = row
            clique_counts = json.loads(clique_counts_json) if clique_counts_json else []

            instance = CliqueProblem(
                graph=g, nodes=n, edge_probability=p, clique_counts=clique_counts
            )

            new_db.save_problem_instance(instance)
            count += 1

            if count > 9000:
                return
