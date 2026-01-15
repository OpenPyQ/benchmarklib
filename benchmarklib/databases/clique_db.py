#
# ============================================================================
# CLIQUE DATABASE
# ============================================================================


import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import networkx as nx

from .. import CliqueProblem, BaseProblem
from benchmarklib.core.types import _ProblemInstance
from .problem_storage import ProblemStorage

logger = logging.getLogger("benchmarklib.databases.clique_db")

# Type variable for problem instances
T = TypeVar("T", bound=_ProblemInstance)


class CliqueDatabase(ProblemStorage):
    """
    Specialized database for Clique problems.

    Provides a clean, intuitive interface for clique problem storage and retrieval.
    """

    DEFAULT_SHARED_PATH = "shared/problems/clique_problems.db"

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False):
        """
        Initialize clique database.

        Args:
            db_path: Path to database. If None, uses shared database.
            read_only: If True, opens in read-only mode.
        """
        if db_path is None:
            db_path = self._get_shared_path()
            read_only = True

        super().__init__(db_path, CliqueProblem, read_only)

    @classmethod
    def shared(cls) -> "CliqueDatabase":
        """Get the shared (read-only) database instance."""
        return cls(db_path=None, read_only=True)

    @classmethod
    def local(cls, path: str) -> "CliqueDatabase":
        """Create or open a local (read-write) database."""
        return cls(db_path=path, read_only=False)

    def _get_shared_path(self) -> str:
        """Get path to shared database."""
        # Try package-relative path first
        package_dir = Path(__file__).parent.parent.parent
        shared_path = package_dir / "shared" / "problems" / "clique.db"

        if not shared_path.exists():
            # Fallback to relative to current directory
            shared_path = Path(self.DEFAULT_SHARED_PATH)

        return str(shared_path)

    def _extract_indexed_fields(self, size_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract indexed fields for clique problems."""
        return {
            "idx_int1": size_metrics.get("num_vertices"),
            "idx_int2": size_metrics.get("num_edges"),
            "idx_int3": size_metrics.get("edge_probability"),
        }

    def _map_filters_to_indices(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Map user-friendly filter names to indexed columns."""
        result = {}

        # Map various filter names to indexed columns
        # idx_int1 = num_vertices
        if "num_vertices" in filters:
            result["idx_int1"] = filters["num_vertices"]
        elif "vertices" in filters:
            result["idx_int1"] = filters["vertices"]
        elif "n" in filters:
            result["idx_int1"] = filters["n"]

        # idx_int2 = num_edges
        if "num_edges" in filters:
            result["idx_int2"] = filters["num_edges"]
        elif "edges" in filters:
            result["idx_int2"] = filters["edges"]

        # idx_int3 = edge_probability
        if "edge_probability" in filters:
            result["idx_int3"] = filters["edge_probability"]
        elif "edge_prob" in filters:
            result["idx_int3"] = filters["edge_prob"]
        elif "p" in filters:
            result["idx_int3"] = filters["p"]

        return result

    # ========== Convenience Methods ==========

    def find_by_size(self, n: int, limit: Optional[int] = None) -> List[CliqueProblem]:
        """Find all graphs with n vertices."""
        return self.find(num_vertices=n, limit=limit)

    def find_by_density(
        self, n: int, p: int, limit: Optional[int] = None
    ) -> List[CliqueProblem]:
        """Find graphs with specific vertex count and edge probability."""
        return self.find(num_vertices=n, edge_probability=p, limit=limit)

    def create_random_graph(
        self, n: int, p: float, compute_cliques: bool = True, save: bool = True
    ) -> CliqueProblem:
        """
        Create a random graph using NetworkX fast_gnp_random_graph.

        Args:
            n: Number of vertices
            p: Edge probability (0.0 to 1.0)
            compute_cliques: Whether to compute clique counts
            save: Whether to save to database

        Returns:
            New clique problem instance
        """
        # Use NetworkX's efficient algorithm
        G = nx.fast_gnp_random_graph(n, p)

        # Convert to CliqueProblem format
        graph_str = ""
        for i in range(n):
            for j in range(i + 1, n):
                graph_str += "1" if G.has_edge(i, j) else "0"

        # Store probability as integer percentage for consistency
        edge_prob_int = int(p * 100)

        problem = CliqueProblem(
            graph=graph_str, nodes=n, edge_probability=edge_prob_int
        )

        if compute_cliques:
            problem.compute_clique_counts()

        if save:
            if self.read_only:
                raise RuntimeError("Cannot save to read-only database")
            self.save(problem)

        return problem

    def create_from_networkx(
        self, G: nx.Graph, edge_probability: Optional[int] = None, save: bool = True
    ) -> CliqueProblem:
        """
        Create a CliqueProblem from a NetworkX graph.

        Args:
            G: NetworkX graph
            edge_probability: Edge probability if known (as integer percentage)
            save: Whether to save to database

        Returns:
            New clique problem instance
        """
        n = G.number_of_nodes()

        # Ensure nodes are labeled 0 to n-1
        if set(G.nodes()) != set(range(n)):
            # Relabel nodes
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)

        # Convert to edge string
        graph_str = ""
        for i in range(n):
            for j in range(i + 1, n):
                graph_str += "1" if G.has_edge(i, j) else "0"

        # Calculate edge probability if not provided
        if edge_probability is None:
            num_edges = G.number_of_edges()
            max_edges = n * (n - 1) // 2
            if max_edges > 0:
                edge_probability = int(100 * num_edges / max_edges)
            else:
                edge_probability = 0

        problem = CliqueProblem(
            graph=graph_str, nodes=n, edge_probability=edge_probability
        )

        problem.compute_clique_counts()

        if save:
            if self.read_only:
                raise RuntimeError("Cannot save to read-only database")
            self.save(problem)

        return problem

    def bulk_create_random(
        self, n_values: List[int], p_values: List[float], graphs_per_config: int = 10
    ) -> int:
        """
        Bulk create random graphs for multiple configurations.

        Args:
            n_values: List of vertex counts
            p_values: List of edge probabilities (0.0 to 1.0)
            graphs_per_config: Number of graphs per (n, p) pair

        Returns:
            Number of graphs created
        """
        if self.read_only:
            raise RuntimeError("Cannot save to read-only database")

        count = 0
        for n in n_values:
            for p in p_values:
                for _ in range(graphs_per_config):
                    self.create_random_graph(n, p, compute_cliques=True, save=True)
                    count += 1

                    if count % 100 == 0:
                        logger.info(f"Created {count} graphs...")

        logger.info(f"Created {count} total graphs")
        return count

    def get_size_distribution(self) -> Dict[int, int]:
        """Get distribution of graph sizes in database."""
        query = """
            SELECT idx_int1 as n, COUNT(*) as count
            FROM problem_instances
            WHERE problem_type = ? AND idx_int1 IS NOT NULL
            GROUP BY idx_int1
            ORDER BY idx_int1
        """

        distribution = {}
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (self.problem_type,))

            for row in cursor:
                distribution[row["n"]] = row["count"]

        return distribution

    def get_density_distribution(self, n: Optional[int] = None) -> Dict[int, int]:
        """
        Get distribution of edge probabilities.

        Args:
            n: If specified, only for graphs with n vertices

        Returns:
            Dictionary mapping edge probability to count
        """
        query = """
            SELECT idx_int3 as p, COUNT(*) as count
            FROM problem_instances
            WHERE problem_type = ?
        """
        params = [self.problem_type]

        if n is not None:
            query += " AND idx_int1 = ?"
            params.append(n)

        query += " GROUP BY idx_int3 ORDER BY idx_int3"

        distribution = {}
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            for row in cursor:
                if row["p"] is not None:
                    distribution[row["p"]] = row["count"]

        return distribution

    def statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        stats = {
            "total": self.count(),
            "by_size": self.get_size_distribution(),
            "database_path": str(self.db_path),
            "read_only": self.read_only,
        }

        # Add density information for each size
        stats["by_size_and_density"] = {}
        for size in stats["by_size"].keys():
            stats["by_size_and_density"][size] = self.get_density_distribution(size)

        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def setup_shared_database(populate: bool = True) -> CliqueDatabase:
    """
    Set up the shared clique database.

    Args:
        populate: Whether to populate with sample problems

    Returns:
        CliqueDatabase instance
    """
    shared_dir = Path("shared/problems")
    shared_dir.mkdir(parents=True, exist_ok=True)

    db_path = shared_dir / "clique_problems.db"
    db = CliqueDatabase(str(db_path), read_only=False)

    if populate and db.count() == 0:
        logger.info("Populating shared database with sample problems...")
        db.bulk_create_random(
            n_values=list(range(3, 21)),
            p_values=[float(i / 101) for i in range(1, 100)],
            graphs_per_config=10,
        )

    logger.info(f"Shared database ready at {db_path}")
    return db


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


# Singleton instance for shared database
_shared_clique_db = None


def clique_db() -> CliqueDatabase:
    """
    Quick access to shared clique database (singleton pattern).

    Examples:
        from benchmarklib.databases import clique_db
        problems = clique_db().find(n=5)
    """
    global _shared_clique_db
    if _shared_clique_db is None:
        _shared_clique_db = CliqueDatabase.shared()
    return _shared_clique_db
