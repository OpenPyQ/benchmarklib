import logging
from typing import Dict, Optional, Union

from .core import *  # noqa: F403
from .core import _BenchmarkDatabase  # Temporary for migration

from .problems import (
    CliqueProblem,
    CliqueTrial,
)

# New synthesis benchmarking imports
from .compilers import (
    ClassiqCompiler,
    CompileType,
    SynthesisBenchmark,
    SynthesisCompiler,
    SynthesisResult,
    SynthesisTrial,
    TruthTableCompiler,
    XAGCompiler,
    compare_compilers,
)

from .runners import (
    BatchQueue,
)

from . import pipeline

from . import algorithms, analysis, databases, pipeline


QUANTUM_BENCHMARKING_LOGGER = __name__


def setup_logging(
    level: Union[int, str] = logging.WARNING,
    module_levels: Optional[Dict[str, Union[int, str]]] = None,
    format_string: Optional[str] = None,
):
    """Setup logging for quantum benchmarking modules only."""
    # Get the main library logger
    main_logger = logging.getLogger(QUANTUM_BENCHMARKING_LOGGER)

    # Clear any existing handlers
    main_logger.handlers.clear()

    # Set level and prevent propagation to root
    main_logger.setLevel(level)
    main_logger.propagate = False

    # Add console handler
    handler = logging.StreamHandler()

    if format_string is None:
        format_string = "%(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    main_logger.addHandler(handler)

    # Set specific module levels if provided
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(
                f"{QUANTUM_BENCHMARKING_LOGGER}.{module_name}"
            )
            module_logger.setLevel(module_level)


# from logging_config import setup_quantum_benchmarking_logging

# Quiet by default (no qiskit noise)
# setup_quantum_benchmarking_logging()

# Or debug specific modules:
# setup_quantum_benchmarking_logging(logging.WARNING, {
#     "grover": logging.DEBUG,
#     "quantum_trials": logging.INFO
# })
