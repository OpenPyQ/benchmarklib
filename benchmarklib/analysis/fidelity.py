import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from classiq.interface.generator.functions.classical_type import (
    CLASSICAL_ATTRIBUTES_TYPES,
)
from qiskit.quantum_info import hellinger_fidelity

from ..compilers import (
    ClassiqCompiler,
    SynthesisCompiler,
    TruthTableCompiler,
    XAGCompiler,
)
from ..core import BenchmarkDatabase

logger = logging.getLogger("benchmarklib.analysis")


def get_complete_fidelity_data(
    db_manager: BenchmarkDatabase, compiler: Optional[SynthesisCompiler] = None
) -> Tuple[List[int], List[int], List[float], List[Optional[float]], List[bool]]:
    """
    Extract complete data showing ALL success rates and fidelity where available.

    Returns:
        n_data: Problem sizes for all trials
        grover_iterations_data: Grover iterations for all trials
        success_rate_data: Success rates for all trials
        fidelity_data: Fidelity data (None where simulation unavailable)
        has_simulation: Boolean mask indicating simulation data availability
    """
    n_data = []
    grover_iterations_data = []
    success_rate_data = []
    fidelity_data = []
    has_simulation = []

    # Get ALL completed trials (not just those with simulation data)
    all_trials = db_manager.find_trials(
        compiler_name=compiler.name if compiler else None, include_pending=False
    )
    if not all_trials:
        logger.warning("Warning: No completed trials found")
        return [], [], [], [], []

    # Group all trials by (n, grover_iterations)
    trial_groups = {}
    for trial in all_trials:
        if trial.is_failed:
            continue

        problem = db_manager.get_problem_instance(trial.instance_id)
        problem_size = problem.get_problem_size()
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

    for (n, grover_iterations), trials in trial_groups.items():
        logger.info(
            f"(n, grover_iterations) = ({n}, {grover_iterations}) \t ({len(trials)} trials)"
        )

        # Calculate success rates for ALL trials in this group
        success_rates = []
        fidelities = []
        has_sim_data = False

        for trial in trials:
            # Success rate can always be calculated from hardware counts
            if trial.counts:
                success_rate = db_manager.calculate_trial_success_rate(trial)
                success_rates.append(success_rate)

                # Fidelity only if simulation data exists
                if trial.simulation_counts:
                    try:
                        fidelity = hellinger_fidelity(
                            trial.counts, trial.simulation_counts
                        )
                        fidelities.append(fidelity)
                        has_sim_data = True
                    except Exception as e:
                        logger.warning(
                            f"Could not calculate fidelity for trial {trial.trial_id}: {e}"
                        )

        if not success_rates:
            continue

        # Store data for this group
        mean_success_rate = np.mean(success_rates)
        mean_fidelity = np.mean(fidelities) if fidelities else None

        n_data.append(n)
        grover_iterations_data.append(grover_iterations)
        success_rate_data.append(mean_success_rate)
        fidelity_data.append(mean_fidelity)
        has_simulation.append(has_sim_data)

        # Logging
        logger.info(f"Mean success rate: {mean_success_rate:.4f}")
        if has_sim_data:
            logger.info(f"Mean Hellinger fidelity: {mean_fidelity:.4f}")
            if mean_fidelity < 0.5:
                logger.info(
                    "⚠️  LOW FIDELITY WARNING: Hardware distribution may be too noisy!"
                )
        else:
            logger.info("No simulation data available (circuit too large)")

    return (
        n_data,
        grover_iterations_data,
        success_rate_data,
        fidelity_data,
        has_simulation,
    )


def plot_enhanced_fidelity_analysis(
    n_data: List[int],
    grover_iterations_data: List[int],
    success_rate_data: List[float],
    fidelity_data: List[Optional[float]],
    has_simulation: List[bool],
    title: str,
    filepath: Optional[str] = None,
    size_label: str = "Problem Size",
) -> None:
    """Create enhanced scatter plot showing complete success rate data and fidelity where available."""
    if not n_data:
        logger.warning("No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Convert to numpy arrays for easier indexing
    n_arr = np.array(n_data)
    grover_arr = np.array(grover_iterations_data)
    success_arr = np.array(success_rate_data)
    fidelity_arr = np.array(fidelity_data, dtype=object)
    has_sim_arr = np.array(has_simulation)

    # ============= LEFT PLOT: Fidelity Analysis =============
    ax1.set_title("Hardware vs Simulation Fidelity", fontsize=14, pad=20)

    # Plot points WITH fidelity data
    has_fidelity_mask = has_sim_arr & (fidelity_arr != None)
    if np.any(has_fidelity_mask):
        fidelity_values = [f for f in fidelity_arr[has_fidelity_mask] if f is not None]
        scatter1 = ax1.scatter(
            n_arr[has_fidelity_mask],
            grover_arr[has_fidelity_mask],
            c=fidelity_values,
            cmap="RdYlGn",
            edgecolors="black",
            alpha=0.75,
            s=450,
            vmin=0,
            vmax=1,
            label="With Simulation Data",
        )

    # Plot points WITHOUT fidelity data as white circles
    no_fidelity_mask = ~has_fidelity_mask
    if np.any(no_fidelity_mask):
        ax1.scatter(
            n_arr[no_fidelity_mask],
            grover_arr[no_fidelity_mask],
            c="white",
            edgecolors="black",
            alpha=0.75,
            s=450,
            linewidth=2,
            label="No Simulation Data",
        )

    # Set ticks
    if n_data:
        ax1.set_xticks(np.arange(min(n_data), max(n_data) + 1, 1))
    if grover_iterations_data:
        ax1.set_yticks(
            np.arange(min(grover_iterations_data), max(grover_iterations_data) + 1, 1)
        )

    ax1.set_xlabel(size_label)
    ax1.set_ylabel("Grover Iterations")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Add colorbar only if we have fidelity data
    if np.any(has_fidelity_mask):
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label(
            "Hellinger Fidelity\n(Hardware vs Simulation)", rotation=270, labelpad=20
        )
        cbar1.ax.axhline(y=0.9, color="green", linestyle="--", linewidth=1, alpha=0.7)
        cbar1.ax.text(
            0.5, 0.92, "Excellent", ha="center", va="bottom", fontsize=8, color="green"
        )
        cbar1.ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
        cbar1.ax.text(
            0.5, 0.52, "Poor", ha="center", va="bottom", fontsize=8, color="red"
        )

    # ============= RIGHT PLOT: Complete Success Rate Data =============
    ax2.set_title("Success Rate (All Available Data)", fontsize=14, pad=20)
    success_max = max(success_rate_data) if success_rate_data else 1

    # Plot ALL success rate data
    scatter2 = ax2.scatter(
        n_arr,
        grover_arr,
        c=success_arr,
        cmap="RdYlGn",
        edgecolors="black",
        alpha=0.75,
        s=450,
        vmin=0,
        vmax=success_max,
    )

    # Add markers to distinguish simulation availability
    # Add subtle markers for points without simulation
    if np.any(no_fidelity_mask):
        ax2.scatter(
            n_arr[no_fidelity_mask],
            grover_arr[no_fidelity_mask],
            facecolors="none",
            edgecolors="white",
            alpha=0.9,
            s=200,
            linewidth=3,
            label="No Simulation Data",
        )

    # Set ticks
    if n_data:
        ax2.set_xticks(np.arange(min(n_data), max(n_data) + 1, 1))
    if grover_iterations_data:
        ax2.set_yticks(
            np.arange(min(grover_iterations_data), max(grover_iterations_data) + 1, 1)
        )

    ax2.set_xlabel(size_label)
    ax2.set_ylabel("Grover Iterations")
    ax2.grid(True, alpha=0.3)

    # Add legend only if there are points without simulation
    if np.any(no_fidelity_mask):
        ax2.legend(loc="upper right")

    # Add colorbar for success rates
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(
        f"Success Rate\nRange: 0 - {success_max:.3f}", rotation=270, labelpad=20
    )

    # ============= Overall formatting =============
    fig.suptitle(title, fontsize=16, y=0.98)

    # Count statistics for footer
    total_points = len(n_data)
    sim_points = np.sum(has_sim_arr)

    fig.text(
        0.5,
        0.02,
        f"Data Coverage: {total_points} total experiments, {sim_points} with simulation data\n"
        f"Low fidelity + high success rate → Suspicious results (uniform noise giving false successes)\n"
        f"High fidelity + high success rate → Genuine quantum performance",
        ha="center",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    if filepath is not None:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_enhanced_fidelity_correlation(
    success_rate_data: List[float],
    fidelity_data: List[Optional[float]],
    has_simulation: List[bool],
    n_data: List[int],
    title: str,
    filepath: Optional[str] = None,
) -> None:
    """Create correlation plot between fidelity and success rate for available data."""

    # Filter to only points with both success rate and fidelity data
    valid_indices = [
        i
        for i, (has_sim, fid) in enumerate(zip(has_simulation, fidelity_data))
        if has_sim and fid is not None
    ]

    if len(valid_indices) < 2:
        logger.warning(
            "Insufficient data points with both success rate and fidelity for correlation analysis"
        )
        return

    valid_fidelity = [fidelity_data[i] for i in valid_indices]
    valid_success = [success_rate_data[i] for i in valid_indices]
    valid_n = [n_data[i] for i in valid_indices]

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        valid_fidelity,
        valid_success,
        c=valid_n,
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
    )

    # Add trend line
    if len(valid_fidelity) > 2:
        z = np.polyfit(valid_fidelity, valid_success, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(valid_fidelity), max(valid_fidelity), 100)
        plt.plot(
            x_trend,
            p(x_trend),
            "r-",
            alpha=0.8,
            linewidth=2,
            label=f"Trend: slope = {z[0]:.3f}",
        )

        correlation = np.corrcoef(valid_fidelity, valid_success)[0, 1]

        if correlation < 0.3:
            interpretation = "⚠️ SUSPICIOUS: Success rate not correlated with fidelity!"
        elif correlation < 0.6:
            interpretation = (
                "🟡 MODERATE: Some correlation between fidelity and success"
            )
        else:
            interpretation = "✅ GOOD: Success rate correlates with fidelity"
    else:
        correlation = 0
        interpretation = "Insufficient data for correlation"

    plt.xlabel("Hellinger Fidelity (Hardware vs Simulation)", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.title(
        f"{title}\nCorrelation: {correlation:.3f} (n={len(valid_indices)} points)",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    if len(valid_fidelity) > 2:
        plt.legend()

    cbar = plt.colorbar(scatter)
    cbar.set_label("Problem Size (n)", rotation=270, labelpad=20)

    # Statistics summary
    total_points = len(success_rate_data)
    excluded_points = total_points - len(valid_indices)

    plt.text(
        0.02,
        0.98,
        f"{interpretation}\n\n"
        f"Analysis includes {len(valid_indices)}/{total_points} points with simulation data\n"
        f"Excluded {excluded_points} points due to missing simulation\n\n"
        f"Expected: Higher fidelity → Higher success rate\n"
        f"If uncorrelated: Success may be from noise, not quantum advantage",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def analyze_fidelity(
    db_manager: BenchmarkDatabase,
    compilers: Optional[List[SynthesisCompiler]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Enhanced fidelity analysis showing complete success rate data and fidelity where available."""
    stats = db_manager.get_statistics()
    print(f"Enhanced Fidelity Analysis for {stats['problem_type']} problems:")
    print(f"  Total instances: {stats['problem_instances']}")
    print(f"  Completed trials: {stats['trials']['completed']}")
    print()

    if compilers is None:
        compilers = [XAGCompiler(), TruthTableCompiler(), ClassiqCompiler()]

    for compiler in compilers:
        logger.info(f"\n=== Analyzing {compiler.name} ===")

        (n_data, grover_iter_data, success_data, fidelity_data, has_sim_data) = (
            get_complete_fidelity_data(db_manager, compiler)
        )

        if not n_data:
            logger.info(f"No data found for {compiler.name}")
            continue

        type_str = compiler.name
        title = f"{stats['problem_type']} Enhanced Fidelity Analysis - {type_str}"

        filepath1 = None
        # filepath2 = None
        if save_dir:
            filepath1 = (
                f"{save_dir}/{stats['problem_type']}_enhanced_fidelity_{type_str}.png"
            )
            # filepath2 = f"{save_dir}/{stats['problem_type']}_enhanced_fidelity_correlation_{type_str}.png"

        size_label = "Problem Size"
        if stats["problem_type"].lower() in ["clique", "graph"]:
            size_label = "Vertices Count"
        elif stats["problem_type"].lower() in ["3sat", "sat", "boolean"]:
            size_label = "Variables Count"

        # Create enhanced plots
        plot_enhanced_fidelity_analysis(
            n_data,
            grover_iter_data,
            success_data,
            fidelity_data,
            has_sim_data,
            title,
            filepath1,
            size_label,
        )

        # plot_enhanced_fidelity_correlation(
        #     success_data,
        #     fidelity_data,
        #     has_sim_data,
        #     n_data,
        #     f"{stats['problem_type']} Enhanced Fidelity vs Success Correlation - {type_str}",
        #     filepath2,
        # )

        # Enhanced analysis summary
        total_points = len(n_data)
        sim_points = sum(has_sim_data)
        valid_fidelity = [
            f
            for f, has_sim in zip(fidelity_data, has_sim_data)
            if has_sim and f is not None
        ]

        mean_success = np.mean(success_data)
        mean_fidelity = np.mean(valid_fidelity) if valid_fidelity else None

        # Calculate correlation only for valid points
        # if len(valid_fidelity) > 1:
        #     valid_success = [
        #         success_data[i]
        #         for i, has_sim in enumerate(has_sim_data)
        #         if has_sim and fidelity_data[i] is not None
        #     ]
        #     correlation = np.corrcoef(valid_fidelity, valid_success)[0, 1]
        # else:
        #     correlation = None

        print(f"\nDetailed Analysis for {compiler.name}:")
        print(f"  Total data points: {total_points}")
        print(
            f"  Points with simulation data: {sim_points} ({sim_points / total_points * 100:.1f}%)"
        )
        print(f"  Average success rate (all points): {mean_success:.4f}")

        if mean_fidelity is not None:
            print(f"  Average fidelity (sim available): {mean_fidelity:.4f}")
        else:
            print("  ⚪ NO FIDELITY DATA: All circuits too large for simulation")
            print("      → Consider alternative validation methods for large circuits")
