from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

from .. import (
    ClassiqCompiler,
    SynthesisCompiler,
    TruthTableCompiler,
    XAGCompiler,
)
from ..core import BenchmarkDatabase


def get_probability_data(
    db_manager: BenchmarkDatabase, compiler: Optional[SynthesisCompiler] = None
) -> Tuple[List[int], List[int], List[float]]:
    """
    Extract success rate data from quantum benchmarking database.

    Args:
        db_manager: BenchmarkDatabase instance
        compile_type: Filter by compilation type (None for all)

    Returns:
        Tuple of (n_data, grover_iterations_data, probability_data)
    """
    n_data = []
    grover_iterations_data = []
    probability_data = []

    # Get all trials, optionally filtered by compile_type
    all_trials = db_manager.find_trials(
        compiler_name=compiler.name if compiler else None,
        include_pending=False,  # Only completed trials
    )

    if not all_trials:
        print("Warning: No completed trials found")
        return [], [], []

    # Group trials by (problem_size, grover_iterations)
    trial_groups = {}

    for trial in all_trials:
        # Get problem instance to determine size
        problem = db_manager.get_problem_instance(trial.instance_id)
        problem_size = problem.get_problem_size()

        # Extract key size metric (adapt based on your problem type)
        # For graph problems: num_vertices, for SAT: num_vars, etc.
        size_keys = list(problem_size.keys())
        n = (
            problem_size.get("num_vertices")
            or problem_size.get("num_vars")
            or problem_size.get(size_keys[0])
        )

        grover_iterations = getattr(trial, "grover_iterations", 0) 

        key = (n, grover_iterations)
        if key not in trial_groups:
            trial_groups[key] = []
        trial_groups[key].append(trial)

    # Process each group
    for (n, grover_iterations), trials in trial_groups.items():
        print(
            f"(n, grover_iterations) = ({n}, {grover_iterations}) \t ({len(trials)} trials)"
        )

        if len(trials) == 0:
            print(
                f"Warning: no results for {n} variables, {grover_iterations} iterations; skipping"
            )
            continue

        success_rates = np.zeros(len(trials))
        expected_success_rates = np.zeros(len(trials))

        for i, trial in enumerate(trials):
            # Use database methods to calculate success rates
            success_rates[i] = db_manager.calculate_trial_success_rate(trial)
            expected_success_rates[i] = (
                db_manager.calculate_trial_expected_success_rate(trial)
            )

        # Avoid division by zero
        valid_expected = expected_success_rates > 0
        if np.any(valid_expected):
            mean_ratio = np.mean(
                success_rates[valid_expected] / expected_success_rates[valid_expected]
            )
            mean_success_rate = np.mean(success_rates[valid_expected])
            mean_expected_rate = np.mean(expected_success_rates[valid_expected])
        else:
            mean_ratio = 0.0

        n_data.append(n)
        grover_iterations_data.append(grover_iterations)
        probability_data.append(mean_ratio)
        print(f"Mean success rate: {mean_success_rate:.4f}")
        print(f"Mean success rate: {mean_expected_rate:.4f}")
        print(f"Mean success rate over expected: {mean_ratio:.4f}")

    return n_data, grover_iterations_data, probability_data


def plot_probability_data(
    n_data: List[int],
    grover_iterations_data: List[int],
    probability_data: List[float],
    title: str,
    filepath: Optional[str] = None,
    size_label: str = "Problem Size",
) -> None:
    """
    Create scatter plot of success rate ratios.

    Args:
        n_data: Problem sizes
        grover_iterations_data: Grover iteration counts
        probability_data: Success rate ratios (actual/expected)
        title: Plot title
        filepath: Optional save path
        size_label: Label for x-axis (problem-specific)
    """
    if not n_data:
        print("No data to plot")
        return

    plt.figure(figsize=(20, 10))


    df = pd.DataFrame({
        "n": n_data,
        "grover_iters": grover_iterations_data,
        "ratio": probability_data,
    })

    grid = df.pivot(index="grover_iters", columns="n", values="ratio")
    grid = grid.sort_index().sort_index(axis=1)

    # set NaN values to white since we do not have a full rectangle of data
    cmap = matplotlib.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="white")

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(
        grid.columns,
        grid.index,
        grid.values,
        cmap=cmap,
        shading='nearest',
        vmin=0,
        vmax=0.2
    )

    #scatter = plt.scatter(
    #    n_data,
    #    grover_iterations_data,
    #    c=probability_data,
    #    cmap="RdYlGn",
    #    edgecolors="black",
    #    alpha=0.75,
    #    marker="s",
    #    s=100,
    #    vmin=0,  # Ensure colormap starts at 0
    #    vmax=0.2#min(1.0, max(probability_data)) if probability_data else 1.0,
    #)

    # Set integer ticks
    if n_data:
        plt.xticks(np.arange(min(n_data), max(n_data) + 1, 1))
    if grover_iterations_data:
        plt.yticks(
            np.arange(min(grover_iterations_data), max(grover_iterations_data) + 1, 1)
        )

    plt.xlabel(size_label)
    plt.ylabel("Grover Iterations")
    if title:
        plt.title(title)

    # Add colorbar with label
    cbar = plt.colorbar()
    cbar.set_label("Success Rate / Expected Success Rate", rotation=270, labelpad=20)

    # Add reference line at ratio = 1.0
    cbar.ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1)

    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def analyze_success_rates(
    db_manager: BenchmarkDatabase,
    compilers: Optional[List[SynthesisCompiler]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Complete analysis workflow for success rates.

    Args:
        db_manager: BenchmarkDatabase instance
        compile_types: List of compilation types to analyze (None for all)
        save_dir: Directory to save plots (None for display only)
    """
    # Get database statistics
    stats = db_manager.get_statistics()
    print(f"Analyzing {stats['problem_type']} problems:")
    print(f"  Total instances: {stats['problem_instances']}")
    print(f"  Total trials: {stats['trials']['total']}")
    print(f"  Completed trials: {stats['trials']['completed']}")
    print()

    if compilers is None:
        # TODO if we make an automatic compiler registry the default could just be the full list of compilers
        compilers = [
            XAGCompiler(),
            TruthTableCompiler(),
            ClassiqCompiler(),
        ]  # Analyze all types together

    for compiler in compilers:
        # Get data
        n_data, grover_iter_data, prob_data = get_probability_data(db_manager, compiler)

        if not n_data:
            continue

        # Create title and filename
        type_str = compiler.name if compiler else "All_Types"
        title = f"{stats['problem_type']} Success Rates - {type_str}"

        filepath = None
        if save_dir:
            filepath = (
                f"{save_dir}/{stats['problem_type']}_success_rates_{type_str}.png"
            )

        # Determine appropriate size label based on problem type
        size_label = "Problem Size"
        if stats["problem_type"].lower() in ["clique", "graph"]:
            size_label = "Vertices Count"
        elif stats["problem_type"].lower() in ["3sat", "sat", "boolean"]:
            size_label = "Variables Count"

        # Create plot
        plot_probability_data(
            n_data, grover_iter_data, prob_data, title, filepath, size_label
        )
