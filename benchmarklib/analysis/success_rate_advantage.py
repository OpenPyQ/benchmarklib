from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .. import (
    ClassiqCompiler,
    SynthesisCompiler,
    TruthTableCompiler,
    XAGCompiler,
)
from ..core import BenchmarkDatabase


def get_quantum_advantage_data(
    db_manager: BenchmarkDatabase, compiler: Optional[SynthesisCompiler] = None
) -> Tuple[List[int], List[int], List[float]]:
    """
    Extract quantum advantage factor data from benchmarking database.

    Quantum advantage factor = (p·N) / k
    where N = 2^n (search space), k = iterations, p = success probability

    Args:
        db_manager: BenchmarkDatabase instance
        compiler: Filter by compilation type (None for all)

    Returns:
        Tuple of (n_data, grover_iterations_data, advantage_factor_data)
    """
    n_data = []
    grover_iterations_data = []
    advantage_factor_data = []

    # Get all completed trials
    all_trials = db_manager.find_trials(
        compiler_name=compiler.name,
        include_pending=False,
    )

    if not all_trials:
        print("Warning: No completed trials found")
        return [], [], []

    # Group trials by (problem_size, grover_iterations)
    trial_groups = {}

    for trial in all_trials:
        # Skip failed trials
        if trial.is_failed:
            continue

        problem = db_manager.get_problem_instance(trial.instance_id)
        n = problem.number_of_input_bits()
        grover_iterations = getattr(trial, "grover_iterations", 0) 

        key = (n, grover_iterations)
        if key not in trial_groups:
            trial_groups[key] = []
        trial_groups[key].append(trial)

    # Process each group
    for (n, k), trials in trial_groups.items():
        if len(trials) == 0:
            continue

        print(f"n={n} bits, k={k} iterations ({len(trials)} trials)")

        # Calculate quantum advantage factor for each trial
        advantage_factors = []
        N = 2**n  # Search space size

        for trial in trials:
            p = db_manager.calculate_trial_success_rate(trial)
            problem = db_manager.get_problem_instance(trial.instance_id)
            M = problem.get_number_of_solutions(trial)

            if p > 0 and k > 0 and M > 0:  # Avoid division by zero
                # Classical expected runtime: N/M
                # Quantum expected runtime: k/p
                # Factor = (Classical/Quantum) = (p*N/M) / k
                factor = (p * N) / (M * k)
                advantage_factors.append(factor)

        if advantage_factors:
            mean_factor = np.mean(advantage_factors)
            n_data.append(n)
            grover_iterations_data.append(k)
            advantage_factor_data.append(mean_factor)
            print(f"  Mean quantum advantage factor: {mean_factor:.2f}")
            print(f"  Quantum advantage {'retained' if mean_factor > 1 else 'lost'}")

    return n_data, grover_iterations_data, advantage_factor_data


def plot_quantum_advantage(
    n_data: List[int],
    grover_iterations_data: List[int],
    advantage_factor_data: List[float],
    title: str = "Quantum Advantage Retention",
    filepath: Optional[str] = None,
) -> None:
    """
    Create scatter plot of quantum advantage factors.

    Args:
        n_data: Problem sizes (number of qubits)
        grover_iterations_data: Grover iteration counts
        advantage_factor_data: Quantum advantage factors
        title: Plot title
        filepath: Optional save path
    """
    if not n_data:
        print("No data to plot")
        return

    plt.figure(figsize=(9, 6))

    # Use log scale for color to better visualize the factor
    # Values > 1 show quantum advantage, < 1 show classical advantage
    scatter = plt.scatter(
        n_data,
        grover_iterations_data,
        c=advantage_factor_data,
        cmap="RdYlGn",
        edgecolors="black",
        alpha=0.75,
        s=300,
        norm=LogNorm(vmin=0.1, vmax=max(10, max(advantage_factor_data))),
    )

    # Set integer ticks
    if n_data:
        plt.xticks(np.arange(min(n_data), max(n_data) + 1, 1))
    if grover_iterations_data:
        plt.yticks(
            np.arange(min(grover_iterations_data), max(grover_iterations_data) + 1, 1)
        )

    plt.xlabel("Number of Qubits (n)")
    plt.ylabel("Grover Iterations (k)")
    plt.title(title)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Quantum Advantage Factor", rotation=270, labelpad=20)

    # Add reference line at factor = 1 (break-even point)
    cbar.ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2)

    # Add text labels on colorbar
    # Get colorbar y-axis limits
    y_min, y_max = cbar.ax.get_ylim()

    # Add "Quantum Advantage" text on green side (upper part)
    cbar.ax.text(
        1.5,
        np.exp((np.log(1.0) + np.log(y_max)) / 2),
        "Quantum Advantage",
        rotation=270,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="darkgreen",
        transform=cbar.ax.get_yaxis_transform(),
    )

    # Add "Quantum Disadvantage" text on red side (lower part)
    cbar.ax.text(
        1.5,
        np.exp((np.log(y_min) + np.log(1.0)) / 2),
        "Quantum Disadvantage",
        rotation=270,
        va="center",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="darkred",
        transform=cbar.ax.get_yaxis_transform(),
    )
    # Add grid
    plt.grid(True, alpha=0.3)

    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def analyze_quantum_advantage(
    db_manager: BenchmarkDatabase,
    compilers: Optional[List[SynthesisCompiler]] = None,
    save_dir: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Analyze quantum advantage retention for benchmarked problems.

    Args:
        db_manager: BenchmarkDatabase instance
        compilers: List of compilation types to analyze (None for all)
        save_dir: Directory to save plots (None for display only)
    """
    stats = db_manager.get_statistics()
    print(f"Analyzing quantum advantage for {stats['problem_type']} problems:")
    print(f"  Total instances: {stats['problem_instances']}")
    print(f"  Completed trials: {stats['trials']['completed']}")
    print()

    if compilers is None:
        compilers = [XAGCompiler(), TruthTableCompiler(), ClassiqCompiler()]

    for compiler in compilers:
        # Get data
        n_data, grover_iter_data, advantage_data = get_quantum_advantage_data(
            db_manager, compiler
        )

        if not n_data:
            continue

        # Create title and filename
        name_str = compiler.name if compiler else "All_Compilers"
        title = f"{stats['problem_type']} - Quantum Advantage Retention ({name_str})"

        filepath = None
        if save_dir:
            filepath = (
                f"{save_dir}/{stats['problem_type']}_quantum_advantage_{name_str}.png"
            )

        # Create plot
        plot_quantum_advantage(
            n_data,
            grover_iter_data,
            advantage_data,
            title,
            filepath,
        )
