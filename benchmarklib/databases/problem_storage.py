"""
Clean Problem Storage System using ProblemInstance

A clean storage layer that works with the existing ProblemInstance interface.
Separates storage concerns from problem logic while maintaining compatibility.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Type, TypeVar, Union

from benchmarklib.core.types import BaseProblem

logger = logging.getLogger("benchmarklib.databases")

# Type variable for problem instances
T = TypeVar("T", bound=BaseProblem)


# ============================================================================
# CORE STORAGE ENGINE
# ============================================================================


class ProblemStorage:
    """
    Clean storage engine for ProblemInstance objects.

    Single responsibility: Efficiently store and retrieve problem instances.
    Works with any ProblemInstance subclass.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        problem_class: Type[BaseProblem],
        read_only: bool = False,
    ):
        """
        Initialize storage engine.

        Args:
            db_path: Path to SQLite database
            problem_class: The ProblemInstance subclass this storage handles
            read_only: If True, opens in read-only mode
        """
        self.db_path = Path(db_path)
        self.problem_class = problem_class
        self.read_only = read_only

        # Get problem type from the class
        dummy = problem_class.__new__(problem_class)
        dummy.instance_id = None
        self.problem_type = dummy.problem_type

        if read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database not found: {db_path}")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        if self.read_only:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")

        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create database schema optimized for ProblemInstance storage."""
        with self._connection() as conn:
            # Main table for problem instances
            conn.execute("""
                CREATE TABLE IF NOT EXISTS problem_instances (
                    instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_type TEXT NOT NULL,
                    problem_data TEXT NOT NULL,
                    size_metrics TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Indexed columns for fast filtering
                    -- Extracted from size_metrics for common queries
                    idx_int1 INTEGER,  -- e.g., num_vertices, num_vars
                    idx_int2 INTEGER,  -- e.g., num_edges, num_clauses
                    idx_int3 INTEGER,  -- e.g., edge_probability, clause_ratio
                    idx_real1 REAL,    -- For floating point metrics
                    idx_text1 TEXT     -- For string-based categorization
                )
            """)

            # Create indices for fast queries
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_problem_type ON problem_instances(problem_type)",
                "CREATE INDEX IF NOT EXISTS idx_idx_int1 ON problem_instances(idx_int1)",
                "CREATE INDEX IF NOT EXISTS idx_idx_int2 ON problem_instances(idx_int2)",
                "CREATE INDEX IF NOT EXISTS idx_idx_int3 ON problem_instances(idx_int3)",
                "CREATE INDEX IF NOT EXISTS idx_composite ON problem_instances(problem_type, idx_int1, idx_int2)",
            ]

            for idx_sql in indices:
                conn.execute(idx_sql)

            conn.commit()

    def save(self, problem: BaseProblem) -> int:
        """
        Save a problem instance to storage.

        Args:
            problem: ProblemInstance to save

        Returns:
            Instance ID

        Raises:
            RuntimeError: If storage is read-only
        """
        if self.read_only:
            raise RuntimeError("Cannot save to read-only storage")

        problem_data = json.dumps(problem.to_dict())
        size_metrics = problem.get_problem_size()
        size_metrics_json = json.dumps(size_metrics)

        # Extract indexed fields for fast filtering
        idx_fields = self._extract_indexed_fields(size_metrics)

        with self._connection() as conn:
            cursor = conn.cursor()

            if problem.instance_id is None:
                cursor.execute(
                    """
                    INSERT INTO problem_instances (
                        problem_type, problem_data, size_metrics,
                        idx_int1, idx_int2, idx_int3, idx_real1, idx_text1
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        self.problem_type,
                        problem_data,
                        size_metrics_json,
                        idx_fields.get("idx_int1"),
                        idx_fields.get("idx_int2"),
                        idx_fields.get("idx_int3"),
                        idx_fields.get("idx_real1"),
                        idx_fields.get("idx_text1"),
                    ),
                )
                problem.instance_id = cursor.lastrowid
            else:
                cursor.execute(
                    """
                    UPDATE problem_instances SET
                        problem_data=?, size_metrics=?,
                        idx_int1=?, idx_int2=?, idx_int3=?, idx_real1=?, idx_text1=?
                    WHERE instance_id=?
                """,
                    (
                        problem_data,
                        size_metrics_json,
                        idx_fields.get("idx_int1"),
                        idx_fields.get("idx_int2"),
                        idx_fields.get("idx_int3"),
                        idx_fields.get("idx_real1"),
                        idx_fields.get("idx_text1"),
                        problem.instance_id,
                    ),
                )

            conn.commit()

        return problem.instance_id

    def get(self, instance_id: int) -> BaseProblem:
        """
        Get a problem instance by ID.

        Args:
            instance_id: Problem instance ID

        Returns:
            Problem instance

        Raises:
            KeyError: If problem not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT problem_data FROM problem_instances WHERE instance_id = ?",
                (instance_id,),
            )
            row = cursor.fetchone()

            if not row:
                raise KeyError(f"Problem instance {instance_id} not found")

            problem_data = json.loads(row["problem_data"])
            return self.problem_class.from_dict(problem_data, instance_id=instance_id)

    def find(self, limit: Optional[int] = None, **filters) -> List[BaseProblem]:
        """
        Find problem instances matching filters.

        Args:
            limit: Maximum number of results
            **filters: Filters based on size_metrics fields

        Returns:
            List of matching problem instances
        """
        query = "SELECT instance_id, problem_data FROM problem_instances WHERE problem_type = ?"
        params = [self.problem_type]

        # Map filters to indexed columns
        idx_mapping = self._map_filters_to_indices(filters)
        for col, val in idx_mapping.items():
            if val is not None:
                query += f" AND {col} = ?"
                params.append(val)

        if limit:
            query += f" LIMIT {limit}"

        problems = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            for row in cursor:
                problem_data = json.loads(row["problem_data"])
                problem = self.problem_class.from_dict(
                    problem_data, instance_id=row["instance_id"]
                )
                problems.append(problem)

        return problems

    def random_sample(self, limit: int = 1, **filters) -> List[BaseProblem]:
        """
        Get random sample of problems.

        Args:
            limit: Number of problems to sample
            **filters: Filters to apply before sampling

        Returns:
            Random sample of problems
        """
        query = "SELECT instance_id, problem_data FROM problem_instances WHERE problem_type = ?"
        params = [self.problem_type]

        idx_mapping = self._map_filters_to_indices(filters)
        for col, val in idx_mapping.items():
            if val is not None:
                query += f" AND {col} = ?"
                params.append(val)

        query += f" ORDER BY RANDOM() LIMIT {limit}"

        problems = []
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            for row in cursor:
                problem_data = json.loads(row["problem_data"])
                problem = self.problem_class.from_dict(
                    problem_data, instance_id=row["instance_id"]
                )
                problems.append(problem)

        return problems

    def count(self, **filters) -> int:
        """Count problems matching filters."""
        query = "SELECT COUNT(*) as count FROM problem_instances WHERE problem_type = ?"
        params = [self.problem_type]

        idx_mapping = self._map_filters_to_indices(filters)
        for col, val in idx_mapping.items():
            if val is not None:
                query += f" AND {col} = ?"
                params.append(val)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()["count"]

    def delete(self, instance_id: int):
        """Delete a problem instance."""
        if self.read_only:
            raise RuntimeError("Cannot delete from read-only storage")

        with self._connection() as conn:
            conn.execute(
                "DELETE FROM problem_instances WHERE instance_id = ?", (instance_id,)
            )
            conn.commit()

    def _extract_indexed_fields(self, size_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract fields from size_metrics for indexing.

        Override in subclasses for problem-specific mappings.
        """
        return {}

    def _map_filters_to_indices(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map user filters to indexed columns.

        Override in subclasses for problem-specific mappings.
        """
        return {}
