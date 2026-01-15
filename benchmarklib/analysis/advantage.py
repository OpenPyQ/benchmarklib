import math
from re import S
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..compilers import (
    ClassiqCompiler,
    SynthesisCompiler,
    TruthTableCompiler,
    XAGCompiler,
)
from ..core import BenchmarkDatabase


def get_oracle_calls_data(
    db_manager: BenchmarkDatabase, compiler: Optional[SynthesisCompiler] = None
) -> Tuple[List[int], List[int], List[float], List[int], List[float]]:
    """
    Extract oracle calls data for quantum advantage analysis.

    Args:
        db_manager: BenchmarkDatabase instance
        compiler: Filter by compilation type (None for all)

    Returns:
        Tuple of (n_data, grover_iterations_data, expected_oracle_calls_data,
                 num_solutions_data, success_probabilities_data)
        where expected_oracle_calls_data contains the total expected oracle calls
        for the noisy quantum strategy
    """
    n_data = []
    grover_iterations_data = []
    expected_oracle_calls_data = []
    num_solutions_data = []
    success_probabilities_data = []

    # Get all completed trials
    all_trials = db_manager.find_trials(
        compiler_name=compiler.name if compiler else None,
        include_pending=False,
    )

    if not all_trials:
        print("Warning: No completed trials found")
        return [], [], [], [], []

    # Group trials by (problem_size, grover_iterations)
    trial_groups = {}

    for trial in all_trials:
        # Skip failed trials
        if trial.is_failed:
            continue

        # Get problem instance to determine size
        problem = db_manager.get_problem_instance(trial.instance_id)
        problem_size = problem.get_problem_size()

        # Extract key size metric (adapt based on problem type)
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
            continue

        # Calculate measured probabilities for all trials in this group
        measured_probs = []
        num_solutions = None

        for trial in trials:
            # Calculate success rate using database method
            success_rate = db_manager.calculate_trial_success_rate(trial)

            if success_rate > 0:  # Only include trials with non-zero success
                measured_probs.append(success_rate)

                # Get number of solutions (should be same for all trials in group)
                if num_solutions is None:
                    problem = trial.get_problem_instance(db_manager)
                    try:
                        num_solutions = problem.get_number_of_solutions(
                            trial
                        )
                    except NotImplementedError:
                        print(
                            f"Warning: get_number_of_solutions not implemented for {problem.problem_type}"
                        )
                        num_solutions = 1  # Default fallback

        if not measured_probs or num_solutions is None:
            print(
                f"Warning: No valid data for n={n}, grover_iterations={grover_iterations}"
            )
            continue

        # Average the measured probabilities
        avg_measured_prob = np.mean(measured_probs)

        # Calculate total expected oracle calls for noisy quantum strategy
        N = 2**n  # Total search space
        M = num_solutions  # Number of solutions

        # Oracle calls per Grover run (actual iterations used, not theoretical optimal)
        oracle_calls_per_run = grover_iterations

        # Expected total oracle calls = (Expected runs) × (Oracle calls per run)
        # Expected runs = 1/P_m
        expected_oracle_calls = (1.0 / avg_measured_prob) * oracle_calls_per_run

        n_data.append(n)
        grover_iterations_data.append(grover_iterations)
        expected_oracle_calls_data.append(expected_oracle_calls)
        num_solutions_data.append(num_solutions)
        success_probabilities_data.append(avg_measured_prob)

        # Calculate theoretical values for comparison
        theoretical_optimal_calls = (math.pi / 4) * math.sqrt(N / M)
        classical_calls = N / M
        breakeven_probability = (math.pi / 4) * math.sqrt(M / N)

        # Calculate iteration efficiency
        iteration_efficiency = (
            grover_iterations / theoretical_optimal_calls
            if theoretical_optimal_calls > 0
            else 0
        )

        print(f"Average P_m: {avg_measured_prob:.4f}")
        print(f"Expected oracle calls (noisy): {expected_oracle_calls:.2f}")
        print(f"Classical oracle calls: {classical_calls:.2f}")
        print(f"Theoretical optimal oracle calls: {theoretical_optimal_calls:.2f}")
        print(
            f"Grover iterations used: {grover_iterations} (vs optimal: {theoretical_optimal_calls:.1f}, ratio: {iteration_efficiency:.2f})"
        )
        print(f"Break-even probability threshold: {breakeven_probability:.4f}")
        print(
            f"Quantum advantage: {'YES' if expected_oracle_calls < classical_calls else 'NO'}"
        )
        print()

    return (
        n_data,
        grover_iterations_data,
        expected_oracle_calls_data,
        num_solutions_data,
        success_probabilities_data,
    )


def plot_quantum_advantage_analysis(
    n_data: List[int],
    grover_iterations_data: List[int],
    expected_oracle_calls_data: List[float],
    num_solutions_data: List[int],
    success_probabilities_data: List[float],
    title: str,
    filepath: Optional[str] = None,
    size_label: str = "Problem Size (n)",
) -> None:
    """
    Create comprehensive quantum advantage analysis with two complementary plots.

    Args:
        n_data: Problem sizes (number of variables)
        grover_iterations_data: Grover iteration counts used
        expected_oracle_calls_data: Expected total oracle calls for noisy strategy
        num_solutions_data: Number of solutions for each problem
        success_probabilities_data: Measured success probabilities
        title: Plot title
        filepath: Optional save path
        size_label: Label for x-axis
    """
    if not n_data:
        print("No data to plot")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Calculate theoretical optimal iterations for each point
    theoretical_optimal_iterations = []
    iteration_ratios = []

    for i in range(len(n_data)):
        N = 2 ** n_data[i]
        M = num_solutions_data[i]
        optimal = (math.pi / 4) * math.sqrt(N / M)
        theoretical_optimal_iterations.append(optimal)

        # Calculate ratio of actual to optimal iterations
        ratio = grover_iterations_data[i] / optimal if optimal > 0 else 0
        iteration_ratios.append(ratio)

    # ============= LEFT PLOT: Quantum Advantage Analysis =============
    ax1.set_title("Quantum Advantage: Oracle Calls Comparison", fontsize=14, pad=20)

    # Calculate quantum advantage factor for each point
    quantum_advantage_factors = []

    for i in range(len(n_data)):
        N = 2 ** n_data[i]
        M = num_solutions_data[i]
        classical_oracle_calls = N / M
        quantum_oracle_calls = expected_oracle_calls_data[i]

        # Quantum advantage factor: > 1 means quantum is better, < 1 means quantum is worse
        advantage_factor = classical_oracle_calls / quantum_oracle_calls
        quantum_advantage_factors.append(advantage_factor)

    # ============= LEFT PLOT: Quantum Advantage Analysis =============
    ax1.set_title("Quantum Advantage: Oracle Calls Comparison", fontsize=14, pad=20)

    # Plot measured performance (expected oracle calls for noisy quantum strategy)
    scatter1 = ax1.scatter(
        n_data,
        expected_oracle_calls_data,
        c=quantum_advantage_factors,
        cmap="RdYlGn",  # Red=bad (factor < 1), Yellow=neutral, Green=good (factor > 1)
        edgecolors="black",
        alpha=0.8,
        s=120,
        label="Noisy Quantum Strategy",
        zorder=4,
        vmin=0.1,  # Show factors from 0.1 to max
        vmax=max(max(quantum_advantage_factors), 2.0)
        if quantum_advantage_factors
        else 2.0,
    )

    # Calculate reference lines
    n_range = np.arange(min(n_data), max(n_data) + 1)

    # Group data by n to get representative M values
    n_to_solutions = {}
    for n, M in zip(n_data, num_solutions_data):
        if n not in n_to_solutions:
            n_to_solutions[n] = []
        n_to_solutions[n].append(M)

    # Calculate theoretical reference lines
    theoretical_optimal = []
    classical_calls = []

    for n in n_range:
        if n in n_to_solutions:
            # Use average number of solutions for this n
            avg_M = np.mean(n_to_solutions[n])
            N = 2**n

            # Theoretical optimal Grover oracle calls: π/4 * sqrt(N/M)
            optimal_calls = (math.pi / 4) * math.sqrt(N / avg_M)
            theoretical_optimal.append(optimal_calls)

            # Classical random search oracle calls: N/M
            # This is also the quantum advantage boundary
            classical_expected = N / avg_M
            classical_calls.append(classical_expected)

        else:
            # Extrapolate using typical solution density
            if n_to_solutions:
                # Estimate solution density from existing data
                densities = [
                    M / (2**n_val)
                    for n_val, M_list in n_to_solutions.items()
                    for M in M_list
                ]
                avg_density = np.mean(densities)
                estimated_M = max(1, avg_density * (2**n))
            else:
                estimated_M = 1  # Worst case fallback

            N = 2**n
            optimal_calls = (math.pi / 4) * math.sqrt(N / estimated_M)
            classical_expected = N / estimated_M

            theoretical_optimal.append(optimal_calls)
            classical_calls.append(classical_expected)

    # Plot reference lines
    ax1.plot(
        n_range,
        theoretical_optimal,
        "g--",
        linewidth=2.5,
        label="Theoretical Optimal Grover",
        alpha=0.9,
        zorder=3,
    )

    ax1.plot(
        n_range,
        classical_calls,
        "r-",
        linewidth=3,
        label="Classical Random Search = Quantum Advantage Boundary",
        alpha=0.9,
        zorder=2,
    )

    # Formatting for left plot
    ax1.set_yscale("log")
    ax1.set_xlabel(size_label, fontsize=12)
    ax1.set_ylabel("Expected Oracle Calls", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="upper left")

    # Add colorbar for quantum advantage factors
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label(
        "Quantum Advantage Factor\n(Classical/Quantum Oracle Calls)",
        rotation=270,
        labelpad=25,
    )

    # Add reference line at advantage factor = 1.0 (break-even)
    cbar1.ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
    cbar1.ax.text(0.5, 1.0 + 0.05, "Break-even", ha="center", va="bottom", fontsize=8)

    # Set reasonable y-axis limits for left plot
    all_values = expected_oracle_calls_data + theoretical_optimal + classical_calls
    if all_values:
        ax1.set_ylim(min(all_values) * 0.3, max(all_values) * 3)

    # Add text annotation explaining the regions
    ax1.text(
        0.02,
        0.98,
        "Green points below red line: Strong quantum advantage\n"
        "Yellow points below red line: Weak quantum advantage\n"
        "Red points above red line: Quantum disadvantage\n"
        "Red line: Classical random search performance",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # ============= RIGHT PLOT: Grover Iterations vs Quantum Advantage =============
    ax2.set_title(
        "Noise Impact: How Iteration Count Affects Quantum Advantage",
        fontsize=14,
        pad=20,
    )

    # Plot actual vs theoretical optimal iterations, colored by quantum advantage
    scatter2 = ax2.scatter(
        theoretical_optimal_iterations,
        grover_iterations_data,
        c=quantum_advantage_factors,
        cmap="RdYlGn",  # Red=losing advantage, Green=retaining advantage
        edgecolors="black",
        alpha=0.8,
        s=120,
        zorder=4,
        vmin=0.1,
        vmax=max(max(quantum_advantage_factors), 2.0)
        if quantum_advantage_factors
        else 2.0,
    )

    # Add diagonal line showing perfect match
    max_iterations = max(
        max(theoretical_optimal_iterations), max(grover_iterations_data)
    )
    min_iterations = min(
        min(theoretical_optimal_iterations), min(grover_iterations_data)
    )
    diagonal_range = np.linspace(min_iterations, max_iterations, 100)
    ax2.plot(
        diagonal_range,
        diagonal_range,
        "k--",
        alpha=0.7,
        linewidth=2,
        label="Theoretical Optimum",
    )

    # Add reference lines for common iteration ratios
    ax2.plot(
        diagonal_range,
        0.5 * diagonal_range,
        "b:",
        alpha=0.6,
        linewidth=1.5,
        label="50% of Optimum",
    )
    ax2.plot(
        diagonal_range,
        2 * diagonal_range,
        "b:",
        alpha=0.6,
        linewidth=1.5,
        label="200% of Optimum",
    )

    # Formatting for right plot
    ax2.set_xlabel(
        "Theoretical Optimal Iterations ($\\frac{\\pi}{4}\\sqrt{\\frac{N}{M}}$)",
        fontsize=12,
    )
    ax2.set_ylabel("Actual Iterations Used", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="upper left")

    # Add colorbar for quantum advantage factors
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(
        "Quantum Advantage Factor\n(Classical/Quantum Oracle Calls)",
        rotation=270,
        labelpad=25,
    )

    # Add reference line at advantage factor = 1.0 (break-even)
    cbar2.ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
    cbar2.ax.text(0.5, 1.0 + 0.05, "Break-even", ha="center", va="bottom", fontsize=8)

    # Add annotation explaining the pattern to look for
    ax2.text(
        0.02,
        0.98,
        "HYPOTHESIS TEST:\n"
        "Red points above diagonal → More iterations lose advantage\n"
        "Green points below diagonal → Fewer iterations retain advantage\n"
        "Pattern shows noise dominates theoretical optimum",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.98)

    # Adjust layout and save
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def analyze_quantum_advantage(
    db_manager: BenchmarkDatabase,
    save_dir: Optional[str] = None,
) -> None:
    """
    Complete quantum advantage analysis workflow.

    This analysis compares the expected oracle calls for noisy quantum strategies
    against classical random search to determine when quantum advantage is lost.

    The key insight: Quantum advantage is lost when P_m ≤ π/4 * sqrt(M/N),
    where P_m is the measured success probability, M is the number of solutions,
    and N is the total search space size.

    Args:
        db_manager: BenchmarkDatabase instance
        save_dir: Directory to save plots (None for display only)
    """
    # Get database statistics
    stats = db_manager.get_statistics()
    print(f"Quantum Advantage Analysis for {stats['problem_type']} problems:")
    print(f"  Total instances: {stats['problem_instances']}")
    print(f"  Total trials: {stats['trials']['total']}")
    print(f"  Completed trials: {stats['trials']['completed']}")
    print()

    # Analyze XAG and CLASSICAL_FUNCTION separately
    compilers = [XAGCompiler(), TruthTableCompiler(), ClassiqCompiler()]

    for compiler in compilers:
        print(f"\n=== Analyzing {compiler.name} ===")

        # Get data
        (n_data, grover_iter_data, oracle_calls_data, solutions_data, prob_data) = (
            get_oracle_calls_data(db_manager, compiler)
        )

        if not n_data:
            print(f"No data found for {compiler.name}")
            continue

        # Create title and filename
        title = (
            f"{stats['problem_type']} Quantum Advantage Analysis - {compiler.name}\n"
            f"Oracle Calls: Noisy Quantum vs Classical Random Search"
        )

        filepath = None
        if save_dir:
            filepath = f"{save_dir}/{stats['problem_type']}_quantum_advantage_{compiler.name}.png"

        # Determine appropriate size label
        size_label = "Problem Size (n)"
        if stats["problem_type"].lower() in ["clique", "graph"]:
            size_label = "Number of Vertices (n)"
        elif stats["problem_type"].lower() in ["3sat", "sat", "boolean"]:
            size_label = "Number of Variables (n)"

        # Create plot
        plot_quantum_advantage_analysis(
            n_data,
            grover_iter_data,
            oracle_calls_data,
            solutions_data,
            prob_data,
            title,
            filepath,
            size_label,
        )

        # Print summary statistics
        print(f"\nSummary for {compiler.name}:")

        # Calculate how many points show quantum advantage
        advantage_count = 0
        total_points = len(n_data)

        # Calculate iteration efficiency statistics
        iteration_ratios = []
        well_optimized_count = 0  # Within 20% of optimal

        for i in range(total_points):
            expected_quantum_calls = oracle_calls_data[i]
            n, M = n_data[i], solutions_data[i]
            N = 2**n
            classical_calls_single = N / M

            if expected_quantum_calls < classical_calls_single:
                advantage_count += 1

            # Calculate iteration efficiency
            optimal_iterations = (math.pi / 4) * math.sqrt(N / M)
            actual_iterations = grover_iter_data[i]
            ratio = (
                actual_iterations / optimal_iterations if optimal_iterations > 0 else 0
            )
            iteration_ratios.append(ratio)

            # Consider "well optimized" if within 80-120% of optimal
            if 0.8 <= ratio <= 1.2:
                well_optimized_count += 1

        advantage_percentage = (
            (advantage_count / total_points) * 100 if total_points > 0 else 0
        )
        well_optimized_percentage = (
            (well_optimized_count / total_points) * 100 if total_points > 0 else 0
        )

        print(f"  ✓ Quantum Advantage Analysis:")
        print(
            f"    - Points with quantum advantage: {advantage_count}/{total_points} ({advantage_percentage:.1f}%)"
        )
        print(f"    - Average success probability: {np.mean(prob_data):.4f}")
        print(
            f"    - Success probability range: {min(prob_data):.4f} - {max(prob_data):.4f}"
        )

        print(f"  ✓ Grover Iterations Analysis:")
        print(
            f"    - Well-optimized iterations (±20%): {well_optimized_count}/{total_points} ({well_optimized_percentage:.1f}%)"
        )
        print(
            f"    - Average iteration ratio (actual/optimal): {np.mean(iteration_ratios):.2f}"
        )
        print(
            f"    - Iteration ratio range: {min(iteration_ratios):.2f} - {max(iteration_ratios):.2f}"
        )

        # Identify if poor iteration choice is hurting quantum advantage
        poor_iterations_losing_advantage = 0
        for i in range(total_points):
            if (
                oracle_calls_data[i] >= (2 ** n_data[i]) / solutions_data[i]
            ):  # Lost advantage
                if (
                    iteration_ratios[i] < 0.8 or iteration_ratios[i] > 1.2
                ):  # Poor iteration choice
                    poor_iterations_losing_advantage += 1

        if poor_iterations_losing_advantage > 0:
            print(
                f"    - Points losing advantage due to poor iteration choice: {poor_iterations_losing_advantage}"
            )
            print(f"      → Consider optimizing Grover iteration counts!")


def plot_noise_penalty_analysis(
    db_manager: BenchmarkDatabase,
    compiler: Optional[SynthesisCompiler] = None,
    filepath: Optional[str] = None,
) -> None:
    """
    Alternative visualization: Direct correlation between iteration count and quantum advantage.

    This creates a scatter plot showing:
    - X: Actual iterations used (not theoretical optimal)
    - Y: Quantum advantage factor (Classical/Quantum oracle calls)
    - Color: Problem size

    This directly tests the hypothesis that more iterations → worse quantum advantage.
    """
    # Get data
    (n_data, grover_iter_data, oracle_calls_data, solutions_data, prob_data) = (
        get_oracle_calls_data(db_manager, compiler)
    )

    if not n_data:
        print("No data to plot")
        return

    # Calculate quantum advantage factors
    quantum_advantage_factors = []
    for i in range(len(n_data)):
        N = 2 ** n_data[i]
        M = solutions_data[i]
        classical_calls = N / M
        quantum_calls = oracle_calls_data[i]
        advantage_factor = classical_calls / quantum_calls
        quantum_advantage_factors.append(advantage_factor)

    plt.figure(figsize=(12, 8))

    # Create scatter plot: Iterations vs Quantum Advantage
    scatter = plt.scatter(
        grover_iter_data,
        quantum_advantage_factors,
        c=n_data,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )

    # Add horizontal line at advantage factor = 1 (break-even)
    plt.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Break-even (Quantum = Classical)",
        alpha=0.8,
    )

    # Add trend line if there are enough points
    if len(grover_iter_data) > 3:
        z = np.polyfit(grover_iter_data, quantum_advantage_factors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(grover_iter_data), max(grover_iter_data), 100)
        plt.plot(
            x_trend,
            p(x_trend),
            "r-",
            alpha=0.8,
            linewidth=2,
            label=f"Trend: slope = {z[0]:.3f}",
        )

    plt.xlabel("Actual Grover Iterations Used", fontsize=12)
    plt.ylabel(
        "Quantum Advantage Factor\n(Classical Oracle Calls / Quantum Oracle Calls)",
        fontsize=12,
    )
    plt.title(
        "Noise Penalty: How Iteration Count Directly Affects Quantum Advantage",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add colorbar for problem size
    cbar = plt.colorbar(scatter)
    cbar.set_label("Problem Size (n)", rotation=270, labelpad=20)

    # Add annotation
    correlation = np.corrcoef(grover_iter_data, quantum_advantage_factors)[0, 1]
    plt.text(
        0.02,
        0.98,
        f"Correlation: {correlation:.3f}\n"
        f"Negative correlation → More iterations hurt performance\n"
        f"Positive correlation → More iterations help performance",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_iteration_efficiency_heatmap(
    db_manager: BenchmarkDatabase,
    compiler: Optional[SynthesisCompiler] = None,
    filepath: Optional[str] = None,
) -> None:
    """
    Alternative visualization: 2D heatmap showing quantum advantage across problem size and iteration count.

    This creates a heatmap with:
    - X: Problem size (n)
    - Y: Grover iterations used
    - Color: Quantum advantage factor

    This shows the "sweet spot" of problem sizes and iteration counts for quantum advantage.
    """
    # Get data
    (n_data, grover_iter_data, oracle_calls_data, solutions_data, prob_data) = (
        get_oracle_calls_data(db_manager, compiler)
    )

    if not n_data:
        print("No data to plot")
        return

    # Calculate quantum advantage factors
    quantum_advantage_factors = []
    for i in range(len(n_data)):
        N = 2 ** n_data[i]
        M = solutions_data[i]
        classical_calls = N / M
        quantum_calls = oracle_calls_data[i]
        advantage_factor = classical_calls / quantum_calls
        quantum_advantage_factors.append(advantage_factor)

    # Create 2D grid for heatmap
    n_unique = sorted(list(set(n_data)))
    iter_unique = sorted(list(set(grover_iter_data)))

    # Create advantage matrix
    advantage_matrix = np.full((len(iter_unique), len(n_unique)), np.nan)

    for i in range(len(n_data)):
        n_idx = n_unique.index(n_data[i])
        iter_idx = iter_unique.index(grover_iter_data[i])
        advantage_matrix[iter_idx, n_idx] = quantum_advantage_factors[i]

    plt.figure(figsize=(12, 8))

    # Create heatmap
    im = plt.imshow(
        advantage_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=0.1,
        vmax=max(quantum_advantage_factors),
        extent=[
            min(n_unique) - 0.5,
            max(n_unique) + 0.5,
            min(iter_unique) - 0.5,
            max(iter_unique) + 0.5,
        ],
        origin="lower",
    )

    plt.colorbar(im, label="Quantum Advantage Factor")
    plt.xlabel("Problem Size (n)", fontsize=12)
    plt.ylabel("Grover Iterations Used", fontsize=12)
    plt.title("Quantum Advantage Landscape: Sweet Spots and Dead Zones", fontsize=14)

    # Add contour line at advantage = 1.0
    CS = plt.contour(
        advantage_matrix,
        levels=[1.0],
        colors="black",
        linewidths=2,
        extent=[
            min(n_unique) - 0.5,
            max(n_unique) + 0.5,
            min(iter_unique) - 0.5,
            max(iter_unique) + 0.5,
        ],
        origin="lower",
    )
    plt.clabel(CS, inline=True, fontsize=10, fmt="Break-even")

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_grover_iterations_analysis(
    db_manager: BenchmarkDatabase,
    compiler: Optional[SynthesisCompiler] = None,
    filepath: Optional[str] = None,
) -> None:
    """
    Create focused analysis of Grover iteration optimization.

    This plot specifically examines how well experimentalists are choosing
    their Grover iteration counts compared to theoretical optimal values.

    Args:
        db_manager: BenchmarkDatabase instance
        compiler: Filter by compilation type
        filepath: Optional save path
    """
    # Get data
    (n_data, grover_iter_data, oracle_calls_data, solutions_data, prob_data) = (
        get_oracle_calls_data(db_manager, compiler)
    )

    if not n_data:
        print("No data to plot")
        return

    # Calculate theoretical optimal iterations and quantum advantage
    theoretical_optimal_iterations = []
    quantum_advantage_factors = []

    for i in range(len(n_data)):
        N = 2 ** n_data[i]
        M = solutions_data[i]
        optimal = (math.pi / 4) * math.sqrt(N / M)
        theoretical_optimal_iterations.append(optimal)

        classical_calls = N / M
        quantum_calls = oracle_calls_data[i]
        advantage_factor = classical_calls / quantum_calls
        quantum_advantage_factors.append(advantage_factor)

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = plt.scatter(
        theoretical_optimal_iterations,
        grover_iter_data,
        c=quantum_advantage_factors,
        cmap="RdYlGn",
        s=100,
        alpha=0.7,
        edgecolors="black",
        vmin=0.1,
        vmax=max(quantum_advantage_factors) if quantum_advantage_factors else 2.0,
    )

    # Add reference lines
    max_iter = max(max(theoretical_optimal_iterations), max(grover_iter_data))
    min_iter = min(min(theoretical_optimal_iterations), min(grover_iter_data))
    iter_range = np.linspace(min_iter, max_iter, 100)

    plt.plot(
        iter_range,
        iter_range,
        "k--",
        linewidth=2,
        label="Theoretical Optimum",
        alpha=0.8,
    )
    plt.plot(
        iter_range, 0.8 * iter_range, "b:", linewidth=1.5, label="±20% Range", alpha=0.6
    )
    plt.plot(iter_range, 1.2 * iter_range, "b:", linewidth=1.5, alpha=0.6)

    # Formatting
    plt.xlabel(
        "Theoretical Optimal Iterations ($\\frac{\\pi}{4}\\sqrt{\\frac{N}{M}}$)",
        fontsize=12,
    )
    plt.ylabel("Actual Iterations Used", fontsize=12)
    plt.title("Grover Iteration Optimization vs Quantum Advantage", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(
        "Quantum Advantage Factor\n(Classical/Quantum Oracle Calls)",
        rotation=270,
        labelpad=25,
    )

    # Add reference line at advantage factor = 1.0
    cbar.ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2)

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
