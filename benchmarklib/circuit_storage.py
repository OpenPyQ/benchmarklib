"""
Circuit Storage Module for Quantum Benchmarking Library

SQLite-based storage system for quantum circuits and metadata,
optimized for machine learning workflows and large-scale benchmarking.
"""

import io
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit, qpy

logger = logging.getLogger("benchmarklib.circuit_storage")


class CircuitStorage:
    """
    SQLite-based storage for quantum circuits and their metadata.

    Circuits are stored as:
    - QPY binary data (Qiskit's native format) in BLOB column
    - Metadata in JSON column for flexible querying
    - Indexed properties for fast filtering

    Thread-safe for parallel benchmarking workflows.
    """

    def __init__(self, db_path: str = "circuit_storage.db"):
        """Initialize circuit storage with SQLite database."""
        self.db_path = db_path
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        """Create database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def _initialize_database(self) -> None:
        """Create database tables and indexes if they don't exist."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Create main circuits table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS circuits (
                    circuit_id TEXT PRIMARY KEY,
                    circuit_type TEXT NOT NULL,
                    problem_id INTEGER,
                    propagated_var INTEGER NOT NULL,
                    qpy_data BLOB NOT NULL,
                    num_qubits INTEGER NOT NULL,
                    depth INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    num_parameters INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata JSON NOT NULL
                )
            """)

            # Create indexes for common queries
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_circuit_type ON circuits(circuit_type)",
                "CREATE INDEX IF NOT EXISTS idx_problem_id ON circuits(problem_id)",
                "CREATE INDEX IF NOT EXISTS idx_propagated_var ON circuits(propagated_var)",
                "CREATE INDEX IF NOT EXISTS idx_circuit_props ON circuits(num_qubits, depth)",
                "CREATE INDEX IF NOT EXISTS idx_created_at ON circuits(created_at)",
                # Composite indexes for complex queries
                "CREATE INDEX IF NOT EXISTS idx_type_problem ON circuits(circuit_type, problem_id)",
                "CREATE INDEX IF NOT EXISTS idx_type_prop ON circuits(circuit_type, propagated_var)",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            # Create view for ML-friendly data export
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS circuit_features AS
                SELECT 
                    circuit_id,
                    circuit_type,
                    problem_id,
                    propagated_var,
                    num_qubits,
                    depth,
                    size,
                    num_parameters,
                    json_extract(metadata, '$.n_vertices') as n_vertices,
                    json_extract(metadata, '$.edge_probability') as edge_probability,
                    json_extract(metadata, '$.clique_size') as clique_size,
                    json_extract(metadata, '$.problem_size') as problem_size,
                    created_at
                FROM circuits
            """)

            conn.commit()
            logger.info(f"Circuit storage initialized: {self.db_path}")

    def generate_circuit_id(
        self, circuit_type: str, problem_id: int, propagated_var: int, **kwargs
    ) -> str:
        """
        Generate unique circuit ID based on circuit properties.

        Args:
            circuit_type: Type of circuit (e.g., "clique", "3sat")
            problem_id: Database ID of the problem instance
            propagated_var: Unit propagation variable (-1 for original)
            **kwargs: Additional identifying parameters

        Returns:
            Unique circuit identifier
        """
        id_parts = [f"{circuit_type}", f"prob{problem_id}", f"prop{propagated_var}"]

        # Add any additional parameters
        for key, value in sorted(kwargs.items()):
            id_parts.append(f"{key}{value}")

        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        id_parts.append(timestamp)

        return "_".join(id_parts)

    def save_circuit(
        self,
        circuit: QuantumCircuit,
        circuit_type: str,
        problem_id: int,
        propagated_var: int,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Save circuit and its metadata to database.

        Args:
            circuit: QuantumCircuit to save
            circuit_type: Type of circuit (e.g., "clique", "3sat")
            problem_id: Database ID of the problem instance
            propagated_var: Unit propagation variable
            metadata: Additional metadata to store

        Returns:
            Circuit ID for retrieval
        """
        # Generate circuit ID
        circuit_id = self.generate_circuit_id(
            circuit_type, problem_id, propagated_var, **metadata.get("id_params", {})
        )

        # Serialize circuit to QPY binary
        qpy_buffer = io.BytesIO()
        qpy.dump(circuit, qpy_buffer)
        qpy_data = qpy_buffer.getvalue()

        # Prepare metadata
        full_metadata = {
            "circuit_id": circuit_id,
            "circuit_type": circuit_type,
            "problem_id": problem_id,
            "propagated_var": propagated_var,
            **metadata,
        }

        # Save to database
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO circuits (
                    circuit_id, circuit_type, problem_id, propagated_var,
                    qpy_data, num_qubits, depth, size, num_parameters,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    circuit_id,
                    circuit_type,
                    problem_id,
                    propagated_var,
                    qpy_data,
                    circuit.num_qubits,
                    circuit.depth(),
                    circuit.size(),
                    circuit.num_parameters,
                    datetime.now().isoformat(),
                    json.dumps(full_metadata),
                ),
            )
            conn.commit()

        logger.debug(f"Saved circuit: {circuit_id}")
        return circuit_id

    def load_circuit(self, circuit_id: str) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """
        Load circuit and metadata by ID.

        Args:
            circuit_id: Circuit identifier

        Returns:
            Tuple of (circuit, metadata)

        Raises:
            ValueError: If circuit not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT qpy_data, metadata FROM circuits 
                WHERE circuit_id = ?
            """,
                (circuit_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Circuit {circuit_id} not found")

            # Deserialize QPY data
            qpy_buffer = io.BytesIO(row["qpy_data"])
            circuits = qpy.load(qpy_buffer)
            circuit = circuits[0]  # QPY returns a list

            # Parse metadata
            metadata = json.loads(row["metadata"])

        return circuit, metadata

    def find_circuits(
        self, limit: Optional[int] = None, offset: int = 0, **filters
    ) -> List[Dict[str, Any]]:
        """
        Find circuits matching filter criteria.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip (for pagination)
            **filters: Key-value pairs to match (supports metadata fields)

        Returns:
            List of circuit records (without QPY data for efficiency)
        """
        query = """
            SELECT 
                circuit_id, circuit_type, problem_id, propagated_var,
                num_qubits, depth, size, num_parameters,
                created_at, metadata
            FROM circuits
            WHERE 1=1
        """
        params = []

        # Build filter conditions
        for key, value in filters.items():
            if key in [
                "circuit_type",
                "problem_id",
                "propagated_var",
                "num_qubits",
                "depth",
                "size",
                "num_parameters",
            ]:
                # Direct column filter
                query += f" AND {key} = ?"
                params.append(value)
            else:
                # JSON metadata filter
                query += f" AND json_extract(metadata, '$.{key}') = ?"
                params.append(value)

        # Add ordering and pagination
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        if offset:
            query += " OFFSET ?"
            params.append(offset)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                record = dict(row)
                record["metadata"] = json.loads(record["metadata"])
                del record["metadata"]["circuit_id"]  # Remove redundancy
                results.append(record)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Total circuits
            cursor.execute("SELECT COUNT(*) FROM circuits")
            total_circuits = cursor.fetchone()[0]

            # By circuit type
            cursor.execute("""
                SELECT circuit_type, COUNT(*) 
                FROM circuits 
                GROUP BY circuit_type
            """)
            by_type = dict(cursor.fetchall())

            # By propagation status
            cursor.execute("""
                SELECT 
                    CASE WHEN propagated_var = -1 THEN 'original' ELSE 'propagated' END as status,
                    COUNT(*)
                FROM circuits
                GROUP BY status
            """)
            by_propagation = dict(cursor.fetchall())

            # Average circuit properties
            cursor.execute("""
                SELECT 
                    AVG(num_qubits) as avg_qubits,
                    AVG(depth) as avg_depth,
                    AVG(size) as avg_size,
                    MIN(depth) as min_depth,
                    MAX(depth) as max_depth
                FROM circuits
            """)
            props = dict(cursor.fetchone())

            # Storage size
            cursor.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            )
            db_size_bytes = cursor.fetchone()[0]

        return {
            "total_circuits": total_circuits,
            "by_type": by_type,
            "by_propagation": by_propagation,
            "circuit_properties": props,
            "storage_size_mb": round(db_size_bytes / (1024 * 1024), 2),
        }

    def export_for_ml(
        self,
        output_file: str = "circuit_dataset.json",
        circuit_type: Optional[str] = None,
        include_qasm: bool = True,
        batch_size: int = 100,
    ) -> None:
        """
        Export circuits and metadata for ML training.

        Args:
            output_file: Output filename
            circuit_type: Filter by circuit type (None for all)
            include_qasm: Include QASM representation (adds processing time)
            batch_size: Number of circuits to process at once
        """
        filters = {}
        if circuit_type:
            filters["circuit_type"] = circuit_type

        # Get total count
        with self._connect() as conn:
            cursor = conn.cursor()
            count_query = "SELECT COUNT(*) FROM circuits"
            if circuit_type:
                count_query += " WHERE circuit_type = ?"
                cursor.execute(count_query, (circuit_type,))
            else:
                cursor.execute(count_query)
            total_count = cursor.fetchone()[0]

        print(f"Exporting {total_count} circuits to {output_file}")

        dataset = []
        offset = 0

        while offset < total_count:
            # Get batch of circuits
            batch_records = self.find_circuits(
                limit=batch_size, offset=offset, **filters
            )

            for record in batch_records:
                # Basic record
                ml_record = {
                    "circuit_id": record["circuit_id"],
                    "features": {
                        "num_qubits": record["num_qubits"],
                        "depth": record["depth"],
                        "size": record["size"],
                        "propagated_var": record["propagated_var"],
                        "circuit_type": record["circuit_type"],
                    },
                    "metadata": record["metadata"],
                }

                # Add QASM if requested
                if include_qasm:
                    try:
                        circuit, _ = self.load_circuit(record["circuit_id"])
                        ml_record["qasm"] = circuit.qasm()
                    except Exception as e:
                        logger.warning(
                            f"Failed to export QASM for {record['circuit_id']}: {e}"
                        )
                        ml_record["qasm"] = None

                dataset.append(ml_record)

            offset += batch_size
            print(f"  Processed {min(offset, total_count)}/{total_count} circuits...")

        # Save dataset
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"✅ Exported {len(dataset)} circuits to {output_file}")

    def create_ml_batch_generator(
        self, batch_size: int = 32, shuffle: bool = True, **filters
    ):
        """
        Generator for ML training that yields batches of circuits.

        Args:
            batch_size: Number of circuits per batch
            shuffle: Whether to randomize order
            **filters: Filters to apply

        Yields:
            Batches of (circuits, features, metadata)
        """
        # Get all matching circuit IDs
        with self._connect() as conn:
            query = "SELECT circuit_id FROM circuits WHERE 1=1"
            params = []

            for key, value in filters.items():
                if key in ["circuit_type", "problem_id", "propagated_var"]:
                    query += f" AND {key} = ?"
                    params.append(value)

            if shuffle:
                query += " ORDER BY RANDOM()"

            cursor = conn.cursor()
            cursor.execute(query, params)
            circuit_ids = [row[0] for row in cursor.fetchall()]

        # Yield batches
        for i in range(0, len(circuit_ids), batch_size):
            batch_ids = circuit_ids[i : i + batch_size]

            circuits = []
            features = []
            metadata = []

            for circuit_id in batch_ids:
                circuit, meta = self.load_circuit(circuit_id)
                circuits.append(circuit)

                # Extract features
                feature_dict = {
                    "num_qubits": circuit.num_qubits,
                    "depth": circuit.depth(),
                    "size": circuit.size(),
                    "propagated_var": meta["propagated_var"],
                }
                features.append(feature_dict)
                metadata.append(meta)

            yield circuits, features, metadata

    def vacuum(self) -> None:
        """Optimize database storage and rebuild indexes."""
        with self._connect() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        logger.info("Database optimized")
