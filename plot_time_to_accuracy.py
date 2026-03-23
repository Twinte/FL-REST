"""
Time-to-Accuracy Plot Generator
=================================
Produces the key paper figure: accuracy vs wall-clock time.

Reads round_metrics.jsonl files from multiple methods and overlays
their convergence curves with real time on the x-axis.

Usage:
  python plot_time_to_accuracy.py \
    fl_logs/comm_experiments/C1_FedPrune_stress_test_bw5mbps_seed42_metrics.jsonl \
    fl_logs/comm_experiments/C2_FIARSE_stress_test_bw5mbps_seed42_metrics.jsonl \
    fl_logs/comm_experiments/C3_FedRolex_stress_test_bw5mbps_seed42_metrics.jsonl

  # Or use glob:
  python plot_time_to_accuracy.py fl_logs/comm_experiments/C*_stress_test_*_metrics.jsonl
"""

import json
import argparse
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np


# Method display config
METHOD_STYLES = {
    "FedPrune": {"color": "#e74c3c", "marker": "o", "label": "FedPrune (Ours)"},
    "FIARSE":  {"color": "#e67e22", "marker": "s", "label": "FIARSE"},
    "FedRolex": {"color": "#3498db", "marker": "^", "label": "FedRolex"},
    "HeteroFL": {"color": "#95a5a6", "marker": "D", "label": "HeteroFL"},
    "FLuID":   {"color": "#9b59b6", "marker": "v", "label": "FLuID"},
}


def detect_method(filepath):
    """Extract method name from filename like C1_FedPrune_stress_test_..."""
    basename = os.path.basename(filepath)
    for method in METHOD_STYLES:
        if method in basename:
            return method
    # Fallback: try to parse from file content
    return basename.split("_")[1] if "_" in basename else "Unknown"


def detect_scenario(filepath):
    """Extract scenario name from filename."""
    basename = os.path.basename(filepath)
    for scenario in ["stress_test", "high_heterogeneity", "mixed_capacities", "standard"]:
        if scenario in basename:
            return scenario
    return "unknown"


def load_metrics(filepath):
    """Load round_metrics.jsonl and return (wall_times, accuracies, round_nums)."""
    rounds = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rounds.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not rounds:
        print(f"  WARNING: No metrics in {filepath}")
        return None, None, None
    
    t0 = rounds[0].get("wall_timestamp", 0)
    
    wall_times = []
    accuracies = []
    round_nums = []
    
    for r in rounds:
        wall_times.append(r.get("wall_timestamp", 0) - t0)
        accuracies.append(r.get("accuracy", 0))
        round_nums.append(r.get("round", 0))
    
    return np.array(wall_times), np.array(accuracies), np.array(round_nums)


def plot_time_to_accuracy(files, output_path="time_to_accuracy.png"):
    """Generate the main figure: accuracy vs wall-clock time."""
    
    # Group files by scenario
    scenarios = {}
    for f in files:
        scenario = detect_scenario(f)
        if scenario not in scenarios:
            scenarios[scenario] = []
        scenarios[scenario].append(f)
    
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(7 * n_scenarios, 5))
    if n_scenarios == 1:
        axes = [axes]
    
    for ax, (scenario, scenario_files) in zip(axes, sorted(scenarios.items())):
        ax.set_title(scenario.replace("_", " ").title(), fontsize=13, fontweight='bold')
        ax.set_xlabel("Wall-Clock Time (seconds)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        for filepath in sorted(scenario_files):
            method = detect_method(filepath)
            wall_times, accuracies, round_nums = load_metrics(filepath)
            
            if wall_times is None:
                continue
            
            style = METHOD_STYLES.get(method, {
                "color": "gray", "marker": "x", "label": method
            })
            
            ax.plot(wall_times, accuracies,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4,
                    linewidth=2,
                    label=style["label"],
                    alpha=0.85)
            
            # Annotate final accuracy
            if len(accuracies) > 0:
                ax.annotate(f'{accuracies[-1]:.1f}%',
                           xy=(wall_times[-1], accuracies[-1]),
                           textcoords="offset points",
                           xytext=(8, 0),
                           fontsize=8,
                           color=style["color"])
        
        ax.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_round_vs_time_comparison(files, output_path="round_vs_time_comparison.png"):
    """
    Side-by-side: accuracy vs round (left) and accuracy vs time (right).
    Shows that methods look similar by round but diverge by wall-clock time.
    """
    # Use only the first scenario found
    scenario_files = files
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: accuracy vs round
    ax1.set_title("Accuracy vs Round", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Communication Round", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: accuracy vs time
    ax2.set_title("Accuracy vs Wall-Clock Time", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Wall-Clock Time (seconds)", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    for filepath in sorted(scenario_files):
        method = detect_method(filepath)
        wall_times, accuracies, round_nums = load_metrics(filepath)
        
        if wall_times is None:
            continue
        
        style = METHOD_STYLES.get(method, {
            "color": "gray", "marker": "x", "label": method
        })
        
        ax1.plot(round_nums, accuracies,
                color=style["color"], marker=style["marker"],
                markersize=4, linewidth=2,
                label=style["label"], alpha=0.85)
        
        ax2.plot(wall_times, accuracies,
                color=style["color"], marker=style["marker"],
                markersize=4, linewidth=2,
                label=style["label"], alpha=0.85)
    
    ax1.legend(fontsize=9, loc='lower right')
    ax2.legend(fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(files):
    """Print a text summary of time-to-accuracy milestones."""
    print(f"\n{'='*70}")
    print("TIME-TO-ACCURACY MILESTONES")
    print(f"{'='*70}")
    
    thresholds = [40, 50, 60, 70]
    
    header = f"{'Method':<12} {'Scenario':<20}"
    for t in thresholds:
        header += f" {'≥'+str(t)+'%':>10}"
    header += f" {'Final':>10} {'Rounds':>8}"
    print(header)
    print("-" * len(header))
    
    for filepath in sorted(files):
        method = detect_method(filepath)
        scenario = detect_scenario(filepath)
        wall_times, accuracies, round_nums = load_metrics(filepath)
        
        if wall_times is None:
            continue
        
        row = f"{method:<12} {scenario:<20}"
        
        for threshold in thresholds:
            reached = np.where(accuracies >= threshold)[0]
            if len(reached) > 0:
                t = wall_times[reached[0]]
                row += f" {t:>9.1f}s"
            else:
                row += f" {'—':>10}"
        
        row += f" {accuracies[-1]:>9.1f}%"
        row += f" {int(round_nums[-1]):>8d}"
        print(row)
    
    # Speedup summary
    print(f"\n{'='*70}")
    print("SPEEDUP: FedPrune vs FIARSE (time to reach threshold)")
    print(f"{'='*70}")
    
    # Group by scenario
    by_scenario = {}
    for filepath in files:
        scenario = detect_scenario(filepath)
        method = detect_method(filepath)
        wall_times, accuracies, _ = load_metrics(filepath)
        if wall_times is not None:
            if scenario not in by_scenario:
                by_scenario[scenario] = {}
            by_scenario[scenario][method] = (wall_times, accuracies)
    
    for scenario, methods in sorted(by_scenario.items()):
        if "FedPrune" not in methods or "FIARSE" not in methods:
            continue
        
        fp_t, fp_a = methods["FedPrune"]
        fi_t, fi_a = methods["FIARSE"]
        
        print(f"\n  {scenario}:")
        for threshold in thresholds:
            fp_reached = np.where(fp_a >= threshold)[0]
            fi_reached = np.where(fi_a >= threshold)[0]
            
            if len(fp_reached) > 0 and len(fi_reached) > 0:
                fp_time = fp_t[fp_reached[0]]
                fi_time = fi_t[fi_reached[0]]
                speedup = fi_time / fp_time if fp_time > 0 else float('inf')
                print(f"    ≥{threshold}%: FedPrune {fp_time:.1f}s vs FIARSE {fi_time:.1f}s → {speedup:.1f}× speedup")
            elif len(fp_reached) > 0:
                print(f"    ≥{threshold}%: FedPrune {fp_t[fp_reached[0]]:.1f}s vs FIARSE never reached")
            elif len(fi_reached) > 0:
                print(f"    ≥{threshold}%: FedPrune never reached vs FIARSE {fi_t[fi_reached[0]]:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Plot time-to-accuracy from round_metrics.jsonl files")
    parser.add_argument("files", nargs="+", help="round_metrics.jsonl files to plot")
    parser.add_argument("--output", type=str, default="time_to_accuracy.png",
                        help="Output filename for main plot")
    parser.add_argument("--comparison", action="store_true",
                        help="Also generate round-vs-time side-by-side comparison")
    args = parser.parse_args()
    
    valid_files = [f for f in args.files if os.path.exists(f)]
    if not valid_files:
        print("No valid metrics files found.")
        sys.exit(1)
    
    print(f"Plotting {len(valid_files)} files...")
    
    # Main time-to-accuracy plot
    plot_time_to_accuracy(valid_files, args.output)
    
    # Optional side-by-side comparison
    if args.comparison:
        base = os.path.splitext(args.output)[0]
        plot_round_vs_time_comparison(valid_files, f"{base}_comparison.png")
    
    # Summary table
    print_summary_table(valid_files)


if __name__ == "__main__":
    main()
