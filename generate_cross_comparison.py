#!/usr/bin/env python3
"""
FedPrune Cross-Comparison Analysis
====================================
Reads experiment_results/<scenario>/<method>_seed<N>/simulation.log
Produces:
  - Mean ± std accuracy tables across seeds
  - Convergence speed (rounds to 60%, 70%)
  - Per-class accuracy fairness (std across classes)
  - Neuron overlap (Jaccard) and coverage (%)
  - Training time per capacity tier
  - Pairwise statistical significance (paired t-test)
  - LaTeX-ready tables for the paper

Usage:
  python generate_cross_comparison.py experiment_results/
"""

import os
import re
import sys
import json
import numpy as np
from collections import defaultdict
from itertools import combinations

# =====================================================================
# LOG PARSING
# =====================================================================

def parse_log(log_path):
    """Extract all metrics from a single simulation log."""
    metrics = {
        "final_accuracy": None,
        "final_loss": None,
        "accuracy_curve": [],    # (round, accuracy) pairs
        "loss_curve": [],        # (round, loss) pairs
        "per_class_acc": [],     # list of [10 floats] per round
        "per_class_std": [],     # std across classes per round
        "neuron_overlap": [],    # avg Jaccard per round
        "tier_overlap": [],      # high-vs-low Jaccard per round
        "neuron_coverage": [],   # cumulative % per round
        "client_neurons": defaultdict(list),  # client_id -> [(round, neurons)]
        "client_train_time": defaultdict(list),  # client_id -> [time_sec]
        "agg_time": [],          # aggregation time per round
        "server_ram": [],        # MB per round
        "server_cpu": [],        # % per round
        "method": None,
        "scenario": None,
        "seed": None,
        "total_rounds": 0,
        "clients": {},           # client_id -> {capacity, profile}
    }

    # Patterns
    p_round = re.compile(r"Round (\d+) Complete\. Accuracy: ([\d.]+)%, Loss: ([\d.]+)")
    p_class = re.compile(r"PER_CLASS_ACC: \[([\d.,\s]+)\] std=([\d.]+)")
    p_overlap = re.compile(r"NEURON_OVERLAP: avg_jaccard=([\d.]+)")
    p_tier = re.compile(r"TIER_OVERLAP: .+ jaccard=([\d.]+)")
    p_coverage = re.compile(r"NEURON_COVERAGE: ([\d.]+)%")
    p_neurons = re.compile(
        r"(client_\d+).*(?:FedPrune|HeteroFL|FedRolex|FIARSE|FLuID|PartialTraining): "
        r"Training (\d+) neurons"
    )
    p_train_time = re.compile(r"(client_\d+).*training complete in ([\d.]+)s", re.IGNORECASE)
    p_agg_time = re.compile(r"PROFILING:.*Time ([\d.]+)s")
    p_ram = re.compile(r"PROFILING:.*RAM ([\d.]+)MB")
    p_cpu = re.compile(r"PROFILING: CPU ([\d.]+)%")
    p_client_reg = re.compile(
        r"(client_\d+): capacity=([\d.]+) profile=(\w+)"
    )
    # Also match the actual server log format: "Registered capacity: client_001 → 0.70"
    p_client_reg_alt = re.compile(
        r"Registered capacity:\s*(client_\d+)\s*\S+\s*([\d.]+)"
    )
    # And server/app.py format: "Client client_001 registered (capacity=0.70)"
    p_client_reg_alt2 = re.compile(
        r"Client (client_\d+) registered \(capacity=([\d.]+)\)"
    )
    p_algo = re.compile(r"CLIENT_ALGO[=:]\s*(\w+)")
    p_strategy = re.compile(r"AGGREGATION_STRATEGY[=:]\s*(\w+)")

    if not os.path.exists(log_path):
        return metrics

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            # Round completion
            m = p_round.search(line)
            if m:
                rnd = int(m.group(1))
                acc = float(m.group(2))
                loss = float(m.group(3))
                metrics["accuracy_curve"].append((rnd, acc))
                metrics["loss_curve"].append((rnd, loss))
                metrics["final_accuracy"] = acc
                metrics["final_loss"] = loss
                metrics["total_rounds"] = rnd + 1

            # Per-class accuracy
            m = p_class.search(line)
            if m:
                accs = [float(x.strip()) for x in m.group(1).split(",")]
                metrics["per_class_acc"].append(accs)
                metrics["per_class_std"].append(float(m.group(2)))

            # Neuron overlap
            m = p_overlap.search(line)
            if m:
                metrics["neuron_overlap"].append(float(m.group(1)))

            # Tier overlap
            m = p_tier.search(line)
            if m:
                metrics["tier_overlap"].append(float(m.group(1)))

            # Neuron coverage
            m = p_coverage.search(line)
            if m:
                metrics["neuron_coverage"].append(float(m.group(1)))

            # Client neurons assigned
            m = p_neurons.search(line)
            if m:
                metrics["client_neurons"][m.group(1)].append(int(m.group(2)))

            # Client training time
            m = p_train_time.search(line)
            if m:
                metrics["client_train_time"][m.group(1)].append(float(m.group(2)))

            # Aggregation time
            m = p_agg_time.search(line)
            if m:
                metrics["agg_time"].append(float(m.group(1)))

            # Server RAM
            m = p_ram.search(line)
            if m:
                metrics["server_ram"].append(float(m.group(1)))

            # Server CPU
            m = p_cpu.search(line)
            if m:
                metrics["server_cpu"].append(float(m.group(1)))

            # Client registration (multiple log formats)
            m = p_client_reg.search(line)
            if m:
                metrics["clients"][m.group(1)] = {
                    "capacity": float(m.group(2)),
                    "profile": m.group(3),
                }
            else:
                # Try alternative format: "Registered capacity: client_001 → 0.70"
                m = p_client_reg_alt.search(line)
                if not m:
                    # Try: "Client client_001 registered (capacity=0.70)"
                    m = p_client_reg_alt2.search(line)
                if m:
                    cid = m.group(1)
                    cap = float(m.group(2))
                    # Infer profile from capacity
                    if cap >= 0.7:
                        profile = "high_perf"
                    elif cap >= 0.4:
                        profile = "mid_perf"
                    else:
                        profile = "low_perf"
                    metrics["clients"][cid] = {
                        "capacity": cap,
                        "profile": profile,
                    }

    return metrics


def rounds_to_threshold(accuracy_curve, threshold):
    """Return the first round where accuracy >= threshold, or None."""
    for rnd, acc in accuracy_curve:
        if acc >= threshold:
            return rnd
    return None


# =====================================================================
# DISCOVERY
# =====================================================================

def discover_experiments(results_dir):
    """
    Scan results_dir/<scenario>/<method>_seed<N>/simulation.log
    Returns: {scenario: {method: {seed: metrics}}}
    """
    experiments = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r"^(\w+)_seed(\d+)$")

    for scenario in sorted(os.listdir(results_dir)):
        scenario_dir = os.path.join(results_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for run_name in sorted(os.listdir(scenario_dir)):
            m = pattern.match(run_name)
            if not m:
                continue
            method = m.group(1)
            seed = int(m.group(2))
            log_path = os.path.join(scenario_dir, run_name, "simulation.log")
            if os.path.exists(log_path):
                metrics = parse_log(log_path)
                if metrics["final_accuracy"] is not None:
                    experiments[scenario][method][seed] = metrics

    return experiments


# =====================================================================
# ANALYSIS
# =====================================================================

def analyze(experiments):
    """Produce all analysis tables from discovered experiments."""
    lines = []

    def p(s=""):
        lines.append(s)

    scenarios = sorted(experiments.keys())
    all_methods = set()
    for s in scenarios:
        all_methods.update(experiments[s].keys())
    methods = sorted(all_methods)

    METHOD_ORDER = ["FedPrune", "HeteroFL", "FedRolex", "FIARSE", "FLuID"]
    methods = [m for m in METHOD_ORDER if m in methods]
    methods += [m for m in all_methods if m not in METHOD_ORDER]

    p("=" * 80)
    p("  FEDPRUNE CROSS-COMPARISON ANALYSIS")
    p("=" * 80)

    # -------------------------------------------------------------------
    # TABLE 1: Final Accuracy (mean ± std)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 1: Final Accuracy (%) — Mean ± Std across seeds")
    p("=" * 80)

    header = f"{'Method':<14}"
    for s in scenarios:
        header += f" {s:>20}"
    p(header)
    p("-" * len(header))

    for method in methods:
        row = f"{method:<14}"
        for scenario in scenarios:
            seeds_data = experiments[scenario].get(method, {})
            accs = [m["final_accuracy"] for m in seeds_data.values()
                    if m["final_accuracy"] is not None]
            if accs:
                row += f" {np.mean(accs):>7.2f}±{np.std(accs):<5.2f}     "
            else:
                row += f" {'N/A':>20}"
        p(row)

    # -------------------------------------------------------------------
    # TABLE 2: Convergence Speed (rounds to 60% and 70%)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 2: Convergence Speed — Rounds to reach threshold (mean)")
    p("=" * 80)

    for threshold in [60.0, 70.0]:
        p(f"\n  Threshold: {threshold:.0f}%")
        header = f"  {'Method':<14}"
        for s in scenarios:
            header += f" {s:>16}"
        p(header)
        p("  " + "-" * (len(header) - 2))

        for method in methods:
            row = f"  {method:<14}"
            for scenario in scenarios:
                seeds_data = experiments[scenario].get(method, {})
                rounds = [rounds_to_threshold(m["accuracy_curve"], threshold)
                          for m in seeds_data.values()]
                rounds = [r for r in rounds if r is not None]
                if rounds:
                    row += f" {np.mean(rounds):>10.1f}     "
                else:
                    row += f" {'>max':>16}"
            p(row)

    # -------------------------------------------------------------------
    # TABLE 3: Per-Class Accuracy Fairness
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 3: Per-Class Accuracy Fairness — Std across classes (lower=fairer)")
    p("=" * 80)

    header = f"{'Method':<14}"
    for s in scenarios:
        header += f" {s:>20}"
    p(header)
    p("-" * len(header))

    for method in methods:
        row = f"{method:<14}"
        for scenario in scenarios:
            seeds_data = experiments[scenario].get(method, {})
            stds = []
            for m in seeds_data.values():
                if m["per_class_std"]:
                    stds.append(m["per_class_std"][-1])  # Last round
            if stds:
                row += f" {np.mean(stds):>8.2f}±{np.std(stds):<5.2f}     "
            else:
                row += f" {'N/A':>20}"
        p(row)

    # -------------------------------------------------------------------
    # TABLE 4: Neuron Overlap (Jaccard)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 4: Neuron Overlap — Avg Jaccard similarity between clients (final round)")
    p("=" * 80)

    header = f"{'Method':<14}"
    for s in scenarios:
        header += f" {s:>20}"
    p(header)
    p("-" * len(header))

    for method in methods:
        row = f"{method:<14}"
        for scenario in scenarios:
            seeds_data = experiments[scenario].get(method, {})
            overlaps = []
            for m in seeds_data.values():
                if m["neuron_overlap"]:
                    overlaps.append(m["neuron_overlap"][-1])
            if overlaps:
                row += f" {np.mean(overlaps):>8.3f}±{np.std(overlaps):<5.3f}    "
            else:
                row += f" {'N/A':>20}"
        p(row)

    # -------------------------------------------------------------------
    # TABLE 5: Neuron Coverage (cumulative %)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 5: Neuron Coverage — % of neurons trained at least once (final)")
    p("=" * 80)

    header = f"{'Method':<14}"
    for s in scenarios:
        header += f" {s:>20}"
    p(header)
    p("-" * len(header))

    for method in methods:
        row = f"{method:<14}"
        for scenario in scenarios:
            seeds_data = experiments[scenario].get(method, {})
            covs = []
            for m in seeds_data.values():
                if m["neuron_coverage"]:
                    covs.append(m["neuron_coverage"][-1])
            if covs:
                row += f" {np.mean(covs):>8.1f}±{np.std(covs):<5.1f}%    "
            else:
                row += f" {'N/A':>20}"
        p(row)

    # -------------------------------------------------------------------
    # TABLE 6: Training Time by Capacity Tier
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 6: Avg Training Time (seconds) — by capacity tier")
    p("=" * 80)

    # Determine tiers present across all experiments
    all_tiers = set()
    for scenario in scenarios:
        for method in methods:
            for m in experiments[scenario].get(method, {}).values():
                for cid, info in m["clients"].items():
                    all_tiers.add(info.get("profile", "unknown"))
    tier_order = ["high_perf", "mid_perf", "low_perf"]
    tiers = [t for t in tier_order if t in all_tiers]
    
    if not tiers:
        # Fallback: group by capacity value
        tiers = ["all"]

    for tier in tiers:
        p(f"\n  Tier: {tier}")
        header = f"  {'Method':<14}"
        for s in scenarios:
            header += f" {s:>16}"
        p(header)
        p("  " + "-" * (len(header) - 2))

        for method in methods:
            row = f"  {method:<14}"
            for scenario in scenarios:
                seeds_data = experiments[scenario].get(method, {})
                times = []
                for m in seeds_data.values():
                    for cid, info in m["clients"].items():
                        cid_profile = info.get("profile", "unknown")
                        match = (tier == "all") or (cid_profile == tier)
                        if match:
                            t = m["client_train_time"].get(cid, [])
                            if t:
                                times.append(np.mean(t))
                if times:
                    row += f" {np.mean(times):>10.1f}s     "
                else:
                    row += f" {'N/A':>16}"
            p(row)

    # -------------------------------------------------------------------
    # TABLE 7: Statistical Significance (paired t-test)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 7: Pairwise Comparison — FedPrune vs each method")
    p("  Δ = FedPrune − Competitor (positive = FedPrune wins)")
    p("  Significance: *** p<0.01, ** p<0.05, * p<0.10, ns p≥0.10")
    p("=" * 80)

    for scenario in scenarios:
        p(f"\n  {scenario.upper()}:")
        fp_seeds = experiments[scenario].get("FedPrune", {})
        if not fp_seeds:
            p("    FedPrune data not found")
            continue

        fp_accs_by_seed = {s: m["final_accuracy"] for s, m in fp_seeds.items()
                          if m["final_accuracy"] is not None}

        for method in methods:
            if method == "FedPrune":
                continue
            comp_seeds = experiments[scenario].get(method, {})
            comp_accs_by_seed = {s: m["final_accuracy"] for s, m in comp_seeds.items()
                                if m["final_accuracy"] is not None}

            # Match seeds
            common_seeds = sorted(set(fp_accs_by_seed) & set(comp_accs_by_seed))
            if len(common_seeds) < 2:
                p(f"    vs {method:<12}: insufficient seeds for t-test")
                continue

            fp_vals = [fp_accs_by_seed[s] for s in common_seeds]
            comp_vals = [comp_accs_by_seed[s] for s in common_seeds]
            diffs = [a - b for a, b in zip(fp_vals, comp_vals)]
            mean_diff = np.mean(diffs)

            # Paired t-test
            from scipy import stats
            if len(common_seeds) >= 2 and np.std(diffs) > 0:
                t_stat, p_val = stats.ttest_rel(fp_vals, comp_vals)
            else:
                p_val = 1.0

            sig = "***" if p_val < 0.01 else (
                  "**" if p_val < 0.05 else (
                  "*" if p_val < 0.10 else "ns"))

            p(f"    vs {method:<12}: Δ={mean_diff:>+6.2f}%  p={p_val:.3f} {sig}")

    # -------------------------------------------------------------------
    # TABLE 8: Server Resource Usage
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("TABLE 8: Server Resources — Peak RAM (MB) and Avg Aggregation Time (ms)")
    p("=" * 80)

    header = f"{'Method':<14} {'Peak RAM (MB)':>14} {'Agg Time (ms)':>14}"
    p(header)
    p("-" * len(header))

    # Use first available scenario for resource comparison
    resource_scenario = scenarios[0] if scenarios else None
    if resource_scenario:
        for method in methods:
            seeds_data = experiments[resource_scenario].get(method, {})
            rams = []
            agg_times = []
            for m in seeds_data.values():
                if m["server_ram"]:
                    rams.append(max(m["server_ram"]))
                if m["agg_time"]:
                    agg_times.append(np.mean(m["agg_time"]) * 1000)  # to ms
            ram_str = f"{np.mean(rams):.0f}" if rams else "N/A"
            agg_str = f"{np.mean(agg_times):.1f}" if agg_times else "N/A"
            p(f"{method:<14} {ram_str:>14} {agg_str:>14}")

    # -------------------------------------------------------------------
    # LATEX TABLE (for paper)
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("LATEX TABLE: Final Accuracy (copy-paste into paper)")
    p("=" * 80)
    p("")
    p("\\begin{table}[t]")
    p("\\centering")
    p("\\caption{Final test accuracy (\\%) across scenarios. Mean$\\pm$std over 3 seeds.}")
    p("\\label{tab:main_results}")
    cols = "l" + "c" * len(scenarios)
    p(f"\\begin{{tabular}}{{{cols}}}")
    p("\\toprule")
    header = "Method"
    for s in scenarios:
        nice_name = s.replace("_", " ").title()
        header += f" & {nice_name}"
    header += " \\\\"
    p(header)
    p("\\midrule")

    for method in methods:
        row = f"{method}"
        for scenario in scenarios:
            seeds_data = experiments[scenario].get(method, {})
            accs = [m["final_accuracy"] for m in seeds_data.values()
                    if m["final_accuracy"] is not None]
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                # Bold if best in column
                all_means = []
                for mm in methods:
                    sd = experiments[scenario].get(mm, {})
                    aa = [x["final_accuracy"] for x in sd.values()
                          if x["final_accuracy"] is not None]
                    if aa:
                        all_means.append(np.mean(aa))
                is_best = abs(mean - max(all_means)) < 0.01 if all_means else False
                if is_best:
                    row += f" & \\textbf{{{mean:.2f}}}$\\pm${std:.2f}"
                else:
                    row += f" & {mean:.2f}$\\pm${std:.2f}"
            else:
                row += " & --"
        row += " \\\\"
        p(row)

    p("\\bottomrule")
    p("\\end{tabular}")
    p("\\end{table}")

    # -------------------------------------------------------------------
    # RANKINGS
    # -------------------------------------------------------------------
    p("\n" + "=" * 80)
    p("RANKINGS PER SCENARIO")
    p("=" * 80)

    for scenario in scenarios:
        p(f"\n  {scenario.upper()}:")
        method_means = []
        for method in methods:
            seeds_data = experiments[scenario].get(method, {})
            accs = [m["final_accuracy"] for m in seeds_data.values()
                    if m["final_accuracy"] is not None]
            if accs:
                method_means.append((method, np.mean(accs), np.std(accs)))
        method_means.sort(key=lambda x: -x[1])
        for rank, (m, mean, std) in enumerate(method_means, 1):
            marker = " ★" if m == "FedPrune" else ""
            p(f"    {rank}. {m:<14} {mean:.2f}±{std:.2f}{marker}")

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_cross_comparison.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    print(f"Scanning {results_dir}...")
    experiments = discover_experiments(results_dir)

    n_scenarios = len(experiments)
    n_runs = sum(
        len(seeds)
        for scenario in experiments.values()
        for seeds in scenario.values()
    )
    print(f"Found {n_scenarios} scenarios, {n_runs} total runs")

    if n_runs == 0:
        print("No completed experiments found.")
        sys.exit(1)

    report = analyze(experiments)
    print(report)

    # Save to file
    output_path = os.path.join(results_dir, "cross_comparison.txt")
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nSaved to {output_path}")

    # Also save raw data as JSON for further analysis
    json_path = os.path.join(results_dir, "raw_results.json")
    raw = {}
    for scenario, methods in experiments.items():
        raw[scenario] = {}
        for method, seeds in methods.items():
            raw[scenario][method] = {}
            for seed, metrics in seeds.items():
                raw[scenario][method][str(seed)] = {
                    "final_accuracy": metrics["final_accuracy"],
                    "final_loss": metrics["final_loss"],
                    "total_rounds": metrics["total_rounds"],
                    "accuracy_curve": metrics["accuracy_curve"],
                    "per_class_std": metrics["per_class_std"][-1] if metrics["per_class_std"] else None,
                    "neuron_overlap": metrics["neuron_overlap"][-1] if metrics["neuron_overlap"] else None,
                    "neuron_coverage": metrics["neuron_coverage"][-1] if metrics["neuron_coverage"] else None,
                }
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()