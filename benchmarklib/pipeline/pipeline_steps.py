import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.singleton import _ctrl_state_to_int
from qiskit_ibm_runtime import IBMBackend

from .synthesis import Synthesizer

logger = logging.getLogger("benchmarklib.pipeline.steps.base")

class PipelineStep(ABC):
    """
    Abstract base class for pipeline transformation steps.

    Each step takes a quantum circuit and returns a transformed version.
    Steps are composable and can be chained together.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this step."""
        pass

    @abstractmethod
    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        """
        Transform the input circuit.

        Pipeline steps must adhere to an implicit contract: The input and target qubits MUST not be moved.
        Perhaps this could be changed in the future, but for now assuming that synthesizer.target_qubit() returns
        the correct qubit index makes our lives much easier.

        Args:
            Synthesizer: Synthesizer contains information about the resulting circuit
            circuit: Input quantum circuit
            **kwargs: Step-specific parameters

        Returns:
            Transformed quantum circuit
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step configuration."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "config": self.get_config(),
        }

    def get_config(self) -> Dict[str, Any]:
        """Override to return step-specific configuration."""
        return {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        """Deserialize step from configuration."""
        raise NotImplementedError("Use StepRegistry.from_dict()")

class StepRegistry:
    """Registry for pipeline step types."""

    _registry: Dict[str, Type[PipelineStep]] = {}

    @classmethod
    def register(cls, step_class: Type[PipelineStep]):
        """Register a step class."""
        cls._registry[step_class.__name__] = step_class
        return step_class

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineStep:
        """Create step from serialized data."""
        step_type = data["type"]
        if step_type not in cls._registry:
            raise ValueError(f"Unknown step type: {step_type}")

        step_class = cls._registry[step_type]
        config = data.get("config", {})

        if hasattr(step_class, "from_config"):
            return step_class.from_config(config)
        else:
            return step_class(**config)

# ============================================================================
# CONCRETE IMPLEMENTATIONS - PIPELINE STEPS
# ============================================================================


@StepRegistry.register
class QiskitTranspile(PipelineStep):
    """Apply Qiskit's transpilation."""

    def __init__(
        self,
        backend: IBMBackend,
        optimization_level: int = 2,
    ):
        self.backend = backend
        self.optimization_level = optimization_level

    @property
    def name(self) -> str:
        return f"QiskitOpt{self.optimization_level}"

    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        from qiskit import transpile

        transpiled = transpile(
            circuit, backend=self.backend, optimization_level=self.optimization_level
        )
        return transpiled

    def get_config(self) -> Dict[str, Any]:
        return {"optimization_level": self.optimization_level}


@StepRegistry.register
class ReplaceCCXwithRCCX(PipelineStep):
    """Apply Swap all CCXs with RCCX which results in better optimization."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        from qiskit.circuit.library import RCCXGate, XGate

        oracle = QuantumCircuit.copy(circuit)

        new_instructions = []
        for instruction in oracle.data:
            op = instruction.operation

            # if the target qubit is in this instruction, we cannot use a relative phase gate
            target_in_inst = any(
                qb._index == synthesizer.target_qubit() for qb in instruction.qubits
            )

            if ("ccx" in op.name or "ccrx" in op.name) and not target_in_inst:
                new_gate = RCCXGate()
                ctrl_state = zip(
                    f"{op.ctrl_state:0{len(instruction.qubits) - 1}b}"[::-1],
                    instruction.qubits,
                )

                ctrl_state_ops = []
                for s, q in ctrl_state:
                    print(f"{s}, {q}")

                    if s == "0":
                        ctrl_state_ops.append((XGate(), [q], ()))

                new_instructions.extend(ctrl_state_ops)
                new_instructions.append(
                    (new_gate, instruction.qubits, instruction.clbits)
                )
                new_instructions.extend(ctrl_state_ops)

                continue

            # If it's not a gate we want to replace, keep the original
            new_instructions.append(instruction)

        # Replace the old circuit data with the new list
        return QuantumCircuit().from_instructions(
            new_instructions, qubits=oracle.qubits, clbits=oracle.clbits
        )

    def get_config(self) -> Dict[str, Any]:
        return {}


@StepRegistry.register
class ReplaceCRXwithCX(PipelineStep):
    """
    Apply Swap all CRXs with CX which might result in better optimization.

    Currently, tweedledum uses Controlled RX(pi) gates instead of Controlled X
    which may be more difficult for qiskit to optimize due to complexity of matrix representations
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        from qiskit.circuit.library import CRXGate, CXGate, RXGate

        oracle = QuantumCircuit.copy(circuit)

        new_instructions = []
        for instruction in oracle.data:
            op = instruction.operation

            # Check if the operation is a ControlledGate
            if isinstance(op, CRXGate) and np.isclose(
                abs(op.base_gate.params[0]), np.pi
            ):
                # Check if its base gate is an RXGate with a parameter of pi
                if isinstance(op.base_gate, RXGate) and np.isclose(
                    abs(op.base_gate.params[0]), np.pi
                ):
                    # Create a new MCXGate with the same control properties
                    new_gate = CXGate(ctrl_state=op.ctrl_state)
                    # Add the new instruction on the same qubits
                    new_instructions.append(
                        (new_gate, instruction.qubits, instruction.clbits)
                    )
                    continue  # Move to the next instruction

            # If it's not a gate we want to replace, keep the original
            new_instructions.append(instruction)

        # Replace the old circuit data with the new list
        return QuantumCircuit().from_instructions(
            new_instructions, qubits=oracle.qubits, clbits=oracle.clbits
        )

    def get_config(self) -> Dict[str, Any]:
        return {}
