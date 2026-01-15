"""
Synthesizer module

Synthesizers take a BaseProblem and return a Quantum Circuit.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional

from qiskit import QuantumCircuit
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input

logger = logging.getLogger("benchmarklib.pipeline.synthesizer")


class Synthesizer(ABC):
    """
    Abstract base class for quantum circuit synthesizers.

    A synthesizer takes a BaseProblem and produces an initial quantum circuit.
    This is the first step in any compilation pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this synthesizer."""
        pass

    @abstractmethod
    def synthesize(self, problem: "BaseProblem", **kwargs) -> QuantumCircuit:
        """
        Synthesize a quantum circuit for the given problem.

        Args:
            problem: Problem instance to synthesize
            **kwargs: Problem-specific parameters (e.g., clique_size)

        Returns:
            Initial quantum circuit (typically a phase-flip oracle)
        """
        pass

    @abstractmethod
    def target_qubit(self) -> Optional[int]:
        """
        Return the index of the target qubit of the oracle
        """

        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize synthesizer configuration."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "config": self.get_config(),
        }

    def get_config(self) -> Dict[str, Any]:
        """Override to return synthesizer-specific configuration."""
        return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Synthesizer":
        """Deserialize synthesizer from configuration."""
        # This would use a registry pattern in practice
        raise NotImplementedError("Use SynthesizerRegistry.from_dict()")


@circuit_input(vertices=lambda n: BitVec(n))
def clique_oracle(n: int, k: int, edges) -> BitVec(1):
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


class SynthesizerRegistry:
    """Registry for synthesizer types to enable serialization."""

    _registry: Dict[str, Type[Synthesizer]] = {}

    @classmethod
    def register(cls, synthesizer_class: Type[Synthesizer]):
        """Register a synthesizer class."""
        cls._registry[synthesizer_class.__name__] = synthesizer_class
        return synthesizer_class

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Synthesizer:
        """Create synthesizer from serialized data."""
        synth_type = data["type"]
        if synth_type not in cls._registry:
            raise ValueError(f"Unknown synthesizer type: {synth_type}")

        synth_class = cls._registry[synth_type]
        config = data.get("config", {})

        # Call class-specific deserialization
        if hasattr(synth_class, "from_config"):
            return synth_class.from_config(config)
        else:
            # Default constructor with config as kwargs
            return synth_class(**config)
