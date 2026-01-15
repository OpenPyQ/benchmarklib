"""
Pipeline-Based Quantum Circuit Compiler System

A flexible, composable architecture for quantum circuit synthesis and optimization.
Each compiler consists of a Synthesizer (creates initial circuit) a series of
PipelineSteps (for backwards compatibility), and a series of transpiler passes (transform/optimize the circuit).
"""
from __future__ import annotations  # needed for type hinting without circular imports

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import hashlib

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit_ibm_runtime import IBMBackend
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from benchmarklib.pipeline.registries import PassManagerFactoryRegistry

from .pipeline_steps import PipelineStep, QiskitTranspile
from .synthesis import Synthesizer
from .config import PipelineConfig

logger = logging.getLogger("benchmarklib.pipeline")

# ============================================================================
# CIRCUIT METRICS
# ============================================================================


@dataclass
class CircuitMetrics:
    """Metrics for a quantum circuit at a specific compilation stage."""

    num_qubits: int
    depth: int
    gate_count: int

    # Gate breakdown
    entangling_count: int = 0
    single_qubit_count: int = 0

    ops: Dict[Instruction, int] = field(default_factory=dict)
    # Custom metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit) -> "CircuitMetrics":
        """Extract metrics from a quantum circuit."""
        from collections import defaultdict

        ops_count = circuit.count_ops()

        counts = defaultdict(int)
        for inst in circuit.data:
            counts[len(inst.qubits)] += 1

        # Count 2 qb gates
        entangling_gate_count = sum(
            [count for n_qubits, count in counts.items() if n_qubits > 1]
        )

        single_qubit_count = counts[1]

        metrics = cls(
            num_qubits=circuit.num_qubits,
            depth=circuit.depth(),
            gate_count=circuit.size(),
            entangling_count=entangling_gate_count,
            single_qubit_count=single_qubit_count,
            ops=ops_count,
        )

        return metrics


# ============================================================================
# PIPELINE COMPILER
# ============================================================================


@dataclass
class CompilationResult:
    """Complete result from a pipeline compilation."""

    compiler_name: str
    success: bool
    total_time: float

    # Metrics at different stages
    synthesis_metrics: Optional[CircuitMetrics] = None  # After synthesis
    high_level_metrics: Optional[CircuitMetrics] = None  # After pipeline steps
    low_level_metrics: Optional[CircuitMetrics] = None  # After transpilation

    # Timing breakdown
    synthesis_time: float = 0.0
    pipeline_times: List[float] = field(default_factory=list)
    transpilation_time: float = 0.0

    # Error tracking
    error_message: Optional[str] = None
    error_stage: Optional[str] = None

    # Final circuit (optional - can be large)
    synthesis_circuit: Optional[QuantumCircuit] = None
    final_circuit: Optional[QuantumCircuit] = None

    # Pipeline configuration for reproducibility
    pipeline_config: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None


class PipelineCompiler:
    """
    A compiler that applies a synthesizer followed by a series of pipeline steps.

    Naming convention: "Synthesizer+Step1+Step2+...+StepN+PassManagerFactory+BackendName+OptionsHash"
    """

    def __init__(
        self,
        synthesizer: Synthesizer,
        steps: Optional[List[PipelineStep]] = None,
        backend: Optional[IBMBackend] = None,
        transpile_options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        pass_manager_factory: Optional[Callable[..., PassManager]] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize a pipeline compiler.

        Args:
            synthesizer: Initial circuit synthesizer
            steps: List of pipeline transformation steps
            backend: Optional backend for transpilation (ignored if pass_manager provided)
            transpile_options: Options for transpilation (ignored if pass_manager provided)
            name: Name of the pipeline (convention "Synthesizer+Step1+Step2+...+StepN")
            pass_manager: Optional custom PassManager for transpilation
            pipeline_config: Alternative option for specifying config via a PipelineConfig object itself (if passed alongside of other params, the other params override)
        """
        self.name = name

        # defaults from pipeline_config if provided
        if pipeline_config is not None:
            synthesizer = synthesizer or pipeline_config.get_synthesizer()
            steps = steps or pipeline_config.get_compiler_steps()
            pass_manager_factory = pass_manager_factory or pipeline_config.pass_manager_factory
            transpile_options = transpile_options or pipeline_config.pass_manager_kwargs
            name = name or pipeline_config.name
            backend = backend or pipeline_config.backend

        self.synthesizer = synthesizer
        self.steps = steps or []
        self.name = name
        self.backend = backend
        self.pass_manager_factory = pass_manager_factory
        self.transpile_options = transpile_options

        if pass_manager_factory is None and backend is not None:
            self.pass_manager_factory = generate_preset_pass_manager
            self.transpile_options = transpile_options or {"optimization_level": 3}

        # Generate name based on components
        if self.name is None:
            name_parts = []
            if synthesizer is not None:
                name_parts.append(synthesizer.name)

            if steps is not None:
                name_parts.extend([step.name for step in steps])

            if pass_manager_factory is not None:
                name_parts.append(
                    PassManagerFactoryRegistry.reverse_lookup(pass_manager_factory)
                )
            if self.backend is not None:
                name_parts.append(self.backend.name)

            if transpile_options is not None:
                options_str = json.dumps(transpile_options, sort_keys=True)
                options_hash = hashlib.md5(options_str.encode('utf-8')).hexdigest()
                name_parts.append(options_hash)
            
            self.name = "+".join(name_parts)

    @property
    def config(self) -> PipelineConfig:
        return PipelineConfig(
            name=self.name,
            synthesizer_config=self.synthesizer.get_config(),
            compiler_config={
                "steps" : [step.to_dict() for step in self.steps]
            },
            pass_manager_factory=self.pass_manager_factory,
            pass_manager_kwargs=self.transpile_options or {},
            backend_name=self.backend.name if self.backend else None
        )
    
    @property
    def pass_manager(self) -> Optional[PassManager]:
        return self.config.get_pass_manager()

    def compile(
        self, problem: BaseProblem, return_intermediate: bool = False, **kwargs
    ) -> CompilationResult:
        """
        Run the complete compilation pipeline.

        Args:
            problem: Problem instance to compile
            return_intermediate: Whether to keep intermediate circuits
            **kwargs: Problem-specific parameters

        Returns:
            CompilationResult with metrics and timings
        """
        result = CompilationResult(
            compiler_name=self.name,
            success=False,
            total_time=0.0,
            pipeline_config=self.to_dict(),
        )

        start_time = time.time()

        try:
            # Stage 1: Synthesis
            synthesis_start = time.time()
            circuit = self.synthesizer.synthesize(problem, **kwargs)
            result.synthesis_time = time.time() - synthesis_start
            result.synthesis_metrics = CircuitMetrics.from_circuit(circuit)
            result.synthesis_circuit = circuit

            # Stage 2: Pipeline steps
            for step in self.steps:
                pipeline_start = time.time()
                circuit = step.transform(self.synthesizer, circuit, **kwargs)
                result.pipeline_times.append(time.time() - pipeline_start)
            result.high_level_metrics = CircuitMetrics.from_circuit(circuit)

            # Stage 3: Transpilation (if backend provided)
            if self.pass_manager:
                transpile_start = time.time()
                transpiled = self.transpile(circuit)
                result.transpilation_time = time.time() - transpile_start
                result.low_level_metrics = CircuitMetrics.from_circuit(transpiled)

                if return_intermediate:
                    result.final_circuit = transpiled
            else:
                # No transpilation - high level is final
                result.low_level_metrics = result.high_level_metrics
                if return_intermediate:
                    result.final_circuit = circuit

            result.success = True

            if hasattr(self.synthesizer, "compilation_artifacts"):
                result.artifacts = self.synthesizer.compilation_artifacts

        except Exception as e:
            result.error_message = str(e)
            result.error_stage = self._get_error_stage(result)
            logger.error(f"Compilation failed at {result.error_stage}: {e}")

        result.total_time = time.time() - start_time
        return result

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile a circuit using the compiler's transpilation settings."""
        pass_manager = self.pass_manager
        if not pass_manager:
            raise ValueError("No transpilation settings defined for this compiler.")
        return pass_manager.run(circuit)

    def _get_error_stage(self, result: CompilationResult) -> str:
        """Determine which stage failed based on timing."""
        if result.synthesis_time == 0:
            return "synthesis"
        elif len(result.pipeline_times) != len(self.steps):
            return "pipeline"
        elif result.transpilation_time == 0 and self.pass_manager:
            return "transpilation"
        return "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize compiler configuration."""
        return self.config.to_dict()

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], backend: Optional[IBMBackend] = None
    ) -> "PipelineCompiler":
        """Deserialize compiler from configuration."""
        return cls.from_config(config=PipelineConfig.from_dict(data))
    
    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineCompiler":
        return cls(pipeline_config=config)


# ============================================================================
# COMPILER FACTORY
# ============================================================================


class CompilerFactory:
    """Factory for creating common compiler configurations."""

    @staticmethod
    def create_basic_compilers() -> List[PipelineCompiler]:
        """Create basic compiler configurations."""
        compilers = []

        # Raw synthesizers
        for synthesizer in [
            XAGSynthesizer(),
            TruthTableSynthesizer(),
            ClassiqSynthesizer(),
        ]:
            compilers.append(PipelineCompiler(synthesizer))

        # With Qiskit optimization
        for level in [1, 2, 3]:
            compilers.append(
                PipelineCompiler(XAGSynthesizer(), [QiskitOptimizationStep(level)])
            )

        return compilers

    @staticmethod
    def create_advanced_compilers() -> List[PipelineCompiler]:
        """Create advanced compiler configurations with multiple steps."""
        compilers = []

        # Multi-step optimization pipeline
        advanced_steps = [
            QiskitTranspile(optimization_level=3),
        ]

        for synthesizer in [XAGSynthesizer(), ClassiqSynthesizer()]:
            compilers.append(PipelineCompiler(synthesizer, advanced_steps))

        return compilers

    @staticmethod
    def from_config_file(
        filepath: str, backend: Optional[IBMBackend] = None
    ) -> List[PipelineCompiler]:
        """Load compiler configurations from JSON file."""
        with open(filepath, "r") as f:
            configs = json.load(f)

        compilers = []
        for config in configs:
            compiler = PipelineCompiler.from_dict(config, backend)
            compilers.append(compiler)

        return compilers


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def example_usage():
    """Example of how to use the pipeline compiler system."""

    # Create a compiler with multiple optimization steps
    compiler = PipelineCompiler(
        synthesizer=XAGSynthesizer(optimize=True),
        steps=[
            CommutationAnalysisStep(),
            CliffordSimplificationStep(),
            QiskitOptimizationStep(optimization_level=3),
        ],
    )

    # The compiler name is automatically generated
    print(f"Compiler name: {compiler.name}")
    # Output: "XAG_OPT+Commute+CliffordSimp+QiskitOpt3"

    # Serialize for storage
    config = compiler.to_dict()
    print(f"Serialized config: {json.dumps(config, indent=2)}")

    # Recreate from config
    restored = PipelineCompiler.from_dict(config)
    assert restored.name == compiler.name

    # Use in benchmarking
    # result = compiler.compile(problem_instance, clique_size=3)
    # print(f"Success: {result.success}")
    # print(f"High-level depth: {result.high_level_metrics.depth}")
    # print(f"Low-level CX count: {result.low_level_metrics.cx_count}")


if __name__ == "__main__":
    example_usage()
