
# Synthesis Benchmarking Module

## Overview

The synthesis benchmarking module provides a simple, extensible framework for comparing different quantum circuit synthesis compilers. It integrates seamlessly with the existing quantum benchmarking infrastructure while maintaining simplicity and readability.

## Key Design Principles

1. **Simplicity First**: Clean interfaces, minimal complexity
2. **Integration**: Works with existing `BaseProblem` and `BenchmarkDatabase`
3. **Extensibility**: Easy to add new compilers
4. **Documentation**: Clear, comprehensive documentation for maintainability

## Architecture

### Core Components

```
SynthesisCompiler (ABC)
    ├── XAGCompiler          # XOR-AND Graph synthesis
    ├── TruthTableCompiler   # PKRM synthesis
    ├── DirectCompiler       # Wrapper for existing oracle()
    └── YourCompiler         # Easy to add new ones

SynthesisBenchmark           # Orchestrates benchmarking
SynthesisResult             # Stores metrics from one run
SynthesisTrial              # Database-compatible trial storage
```

### How It Works

1. **Problem Instance** → **Compiler** → **Phase-Flip Oracle**
2. Metrics are extracted (qubits, depth, time, gates)
3. Results are stored in the existing database
4. Analysis tools compare compiler performance

## Quick Start

### 1. Implement a Compiler

```python
from benchmarklib.synthesis import SynthesisCompiler
from qiskit import QuantumCircuit

class MyCompiler(SynthesisCompiler):
    @property
    def name(self) -> str:
        return "MY_COMPILER"
    
    def compile(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
        # Your synthesis logic here
        n = problem.number_of_input_bits()
        oracle = QuantumCircuit(n + 1)
        
        # IMPORTANT: Must return a PHASE-FLIP oracle
        # Setup phase flip on ancilla (last qubit)
        oracle.x(n)
        oracle.h(n)
        
        # Your synthesis algorithm here
        # ...
        
        oracle.h(n)
        oracle.x(n)
        return oracle
```

### 2. Run Benchmarks

```python
from benchmarklib import BenchmarkDatabase, SynthesisBenchmark
from benchmarklib.clique import CliqueProblem, CliqueTrial

# Setup database
db = BenchmarkDatabase("synthesis.db", CliqueProblem, CliqueTrial)

# Setup compilers
compilers = [
    MyCompiler(),
    XAGCompiler(optimize_xag=True),
    TruthTableCompiler()
]

# Create benchmark
benchmark = SynthesisBenchmark(db, compilers)

# Get problems and run
problems = db.find_problem_instances(size_filters={"num_vertices": 5})
results = benchmark.run_benchmarks(problems, clique_size=3)

# Display results
benchmark.print_summary(results)
```

### 3. Compare Results

```python
from benchmarklib.synthesis import compare_compilers

# Compare stored results
compare_compilers(
    db,
    compiler_names=["MY_COMPILER", "XAG_BATCHER_OPT"],
    problem_filters={"num_vertices": 5},
    clique_size=3
)
```

## Integration with Hardware Benchmarking

The synthesis module integrates with hardware benchmarking through shared `BaseProblem` objects:

```python
# Same problem, different benchmarks
problem = CliqueProblem(graph="1101", nodes=3)

# Synthesis benchmark
synthesis_result = my_compiler.compile(problem, clique_size=2)

# Hardware benchmark (existing code)
oracle = problem.oracle(CompileType.XAG, clique_size=2)
grover_runner.run_grover_benchmark(problem, CompileType.XAG, iterations=2)
```

## Database Schema

Synthesis trials are stored in the same `trials` table as hardware benchmarks:

- `compile_type`: Set to `DIRECT` for synthesis benchmarks
- `job_id`: Format `SYNTHESIS_{compiler_name}`
- `counts`: Stores synthesis metrics as JSON
- Trial parameters stored normally

This allows unified queries across synthesis and hardware results.

## Metrics Collected

Each synthesis run automatically collects:

- **Success/Failure**: Did compilation succeed?
- **Synthesis Time**: End-to-end compilation time
- **Circuit Metrics**:
  - Number of qubits
  - Circuit depth
  - Total gate count
  - CNOT/CX count
  - Single-qubit gate count
- **Error Messages**: If compilation fails

## Adding New Problem Types

When adding support for new problem types to a compiler:

```python
def compile(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
    if isinstance(problem, CliqueProblem):
        return self._compile_clique(problem, **kwargs)
    elif isinstance(problem, SATInst):  # Your new problem type
        return self._compile_sat(problem, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported: {problem.problem_type}")

def _compile_sat(self, problem: SATInst, **kwargs) -> QuantumCircuit:
    # SAT-specific synthesis logic
    pass
```

## Best Practices

### 1. Always Return Phase-Flip Oracles
The framework expects phase-flip oracles (marks solutions with negative phase), not bit-flip oracles.

### 2. Handle Errors Gracefully
```python
def compile(self, problem, **kwargs):
    try:
        # Synthesis logic
        return oracle
    except MemoryError:
        raise MemoryError(f"Problem too large: {problem.get_problem_size()}")
    except Exception as e:
        raise RuntimeError(f"Synthesis failed: {e}")
```

### 3. Document Compiler Parameters
```python
def __init__(self, optimize: bool = True, max_depth: int = 1000):
    """
    Args:
        optimize: Whether to apply optimization passes
        max_depth: Maximum allowed circuit depth
    """
    self.optimize = optimize
    self.max_depth = max_depth
```

### 4. Use Logging
```python
import logging
logger = logging.getLogger("benchmarklib.synthesis.compilers")

def compile(self, problem, **kwargs):
    logger.debug(f"Compiling {problem} with {self.name}")
    # ...
    logger.info(f"Compiled to {oracle.num_qubits} qubits")
```

## Advanced Usage

### Custom Metrics

Add extra metrics through `SynthesisResult.extra_metrics`:

```python
result = SynthesisResult(
    compiler_name=self.name,
    success=True,
    synthesis_time=1.23,
    extra_metrics={
        "optimization_passes": 3,
        "memory_usage_mb": 456,
        "custom_metric": value
    }
)
```

### Batch Processing

Process multiple parameter configurations:

```python
for clique_size in range(2, 6):
    results = benchmark.run_benchmarks(
        problems,
        clique_size=clique_size,
        skip_existing=True
    )
```

### Parallel Compilation

While not built-in, you can parallelize:

```python
from concurrent.futures import ProcessPoolExecutor

def compile_parallel(compiler, problems, **kwargs):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(compiler.compile, p, **kwargs) 
            for p in problems
        ]
        return [f.result() for f in futures]
```

## Troubleshooting

### Q: Compiler runs out of memory on large problems
**A:** Implement size limits in your compiler:
```python
def compile(self, problem, **kwargs):
    if problem.number_of_input_bits() > 20:
        raise ValueError("Problem too large for this compiler")
```

### Q: How to compare with existing CompileType.XAG?
**A:** Use the `DirectCompiler` wrapper:
```python
compilers = [
    MyNewCompiler(),
    DirectCompiler("XAG"),  # Uses existing oracle() method
]
```

### Q: Database grows too large
**A:** Only failed trials and successful metrics are stored, not full circuits. Use `skip_existing=True` to avoid duplicates.

## Future Extensions

Potential improvements (keeping simplicity in mind):

1. **Caching**: Cache compiled oracles for repeated use
2. **Parallel Support**: Built-in parallel compilation
3. **Visualization**: Plot compiler scaling trends
4. **Export**: Export results to CSV/JSON for external analysis

## Example Output

```
SYNTHESIS BENCHMARK SUMMARY
============================================================

XAG_BATCHER_OPT:
----------------------------------------
  Success Rate: 10/10 (100.0%)
  Synthesis Time: 0.234s ± 0.045s
  Avg Qubits: 12.3 ± 2.1
  Avg Depth: 156.7 ± 23.4
  Avg Gates: 298.5 ± 45.2
  Avg CX Gates: 89.2 ± 15.3

TRUTH_TABLE_BATCHER:
----------------------------------------
  Success Rate: 8/10 (80.0%)
  Synthesis Time: 1.567s ± 0.234s
  Avg Qubits: 8.0 ± 0.0
  Avg Depth: 234.5 ± 34.2
  Avg Gates: 456.3 ± 67.8
  Avg CX Gates: 123.4 ± 21.5
  Failures: 2
    Example error: MemoryError: Truth table too large...

============================================================
```
