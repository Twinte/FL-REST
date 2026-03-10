"""
FL-REST Simulation Report Generator
====================================
Parses Docker simulation logs and generates a multi-page PDF report.

Auto-detects FedPrune-specific data and conditionally adds extra pages.
Works unchanged for FedAvg / FedProx / Scaffold / etc.

Pages (always present):
  1. Summary (rounds, accuracy, loss, server stats)
  2. Convergence curves (accuracy + loss)
  3. Server resource profiling (aggregation time + RAM)

Pages (FedPrune only - auto-detected):
  4. Neuron Allocation per Client (shows heterogeneous workloads)
  5. Extraction Dynamics (imp_frac + capacity std over rounds)
  6. Per-Client Training Time (efficiency comparison across devices)
  7. Per-Client Resource Usage (GPU + RAM)

Usage:
  python generate_report.py                           # Auto-detect latest log
  python generate_report.py path/to/simulation.log    # Specific log file
"""

import os
import re
import sys
import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# =============================================================================
# Log Discovery
# =============================================================================

def find_latest_log(log_dir="../fl_logs"):
    """Finds the most recent simulation_*.log file."""
    search_pattern = os.path.join(log_dir, "simulation_*.log")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print(f"❌ No log files found in {log_dir}")
        sys.exit(1)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


# =============================================================================
# Log Parser
# =============================================================================

def parse_fl_log(log_path):
    """
    Reads the unstructured Docker log and extracts key metrics using Regex.

    Returns:
        df_eval:       Global accuracy/loss per round
        df_prof:       Server CPU/RAM/time per round
        fedprune_data: Dict with FedPrune-specific DataFrames (empty if not FedPrune)
    """
    print(f"📄 Parsing log file: {log_path}")

    rounds_data = []
    profiling_data = []

    # --- FedPrune accumulators ---
    fedprune_neuron_data = []
    fedprune_capacity_data = {}
    fedprune_extraction_data = []
    fedprune_client_training = []

    # =================================================================
    # Regex Patterns
    # =================================================================

    # --- Generic ---
    eval_pattern = re.compile(
        r"Round (\d+) Complete\. Accuracy: ([\d.]+)%, Loss: ([\d.]+)")
    prof_pattern = re.compile(
        r"PROFILING: CPU ([\d.]+)%, RAM ([\d.]+)MB, Time ([\d.]+)s")

    # --- FedPrune Server-side ---
    # "Registered capacity: client_001 → 0.70 (3 clients total)"   (new logs)
    capacity_pattern = re.compile(
        r"Registered capacity:\s*(client_\d+)\s*\u2192\s*([\d.]+)")

    # "client_001: ratio=0.50, neurons=314"
    neuron_pattern = re.compile(
        r"(client_\d+): ratio=([\d.]+), neurons=(\d+)")

    # "Capacity std=0.141 → imp_frac=0.64"
    extraction_pattern = re.compile(
        r"Capacity std=([\d.]+)\s*\u2192\s*imp_frac=([\d.]+)")

    # "FedPrune: Computing indices for 5 clients (round 3)"
    indices_round_pattern = re.compile(
        r"Computing indices for \d+ clients \(round (\d+)\)")

    # --- FedPrune Client-side ---
    # "FedPrune: Training 314 neurons across 6 layers"  (or HeteroFL, FedRolex)
    client_neurons_pattern = re.compile(
        r"(client_\d+).*(?:FedPrune|HeteroFL|FedRolex|PartialTraining): Training (\d+) neurons across (\d+) layers")

    # "Training completed: 10445 samples in 2.21s"
    client_train_pattern = re.compile(
        r"(client_\d+).*Training completed: (\d+) samples in ([\d.]+)s")

    # "Resource usage - GPU: 45.38MB, RAM: 1451.13MB"
    client_resource_pattern = re.compile(
        r"(client_\d+).*Resource usage - GPU: ([\d.]+)MB, RAM: ([\d.]+)MB")

    # "[Device] Applied profile: high_perf (capacity=0.7)"
    device_profile_pattern = re.compile(
        r"(client_\d+).*\[Device\] Applied profile:\s*(\w+)")
    # "[Network] Applied profile: slow"
    network_profile_pattern = re.compile(
        r"(client_\d+).*\[Network\] Applied profile:\s*(\w+)")

    # =================================================================
    # State tracking
    # =================================================================
    current_round = 0
    current_indices_round = 0
    client_profiles = {}

    # Per-round client accumulators (flushed on each eval line)
    client_neurons_buf = {}
    client_train_buf = {}
    client_resource_buf = {}

    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:

            # --- Generic: Evaluation ---
            m = eval_pattern.search(line)
            if m:
                current_round = int(m.group(1))
                rounds_data.append({
                    "Round": current_round,
                    "Accuracy": float(m.group(2)),
                    "Loss": float(m.group(3)),
                })
                # Flush client buffers for this round
                _flush_client_round(
                    fedprune_client_training, current_round,
                    fedprune_capacity_data,
                    client_neurons_buf, client_train_buf, client_resource_buf)
                client_neurons_buf = {}
                client_train_buf = {}
                client_resource_buf = {}
                continue

            # --- Generic: Server profiling ---
            m = prof_pattern.search(line)
            if m:
                profiling_data.append({
                    "Round": current_round,
                    "CPU_Usage_%": float(m.group(1)),
                    "RAM_Usage_MB": float(m.group(2)),
                    "Aggregation_Time_s": float(m.group(3)),
                })
                continue

            # --- FedPrune: Capacity registration (new-style logs) ---
            m = capacity_pattern.search(line)
            if m:
                fedprune_capacity_data[m.group(1)] = float(m.group(2))
                continue

            # --- FedPrune: Index computation round ---
            m = indices_round_pattern.search(line)
            if m:
                current_indices_round = int(m.group(1))
                continue

            # --- FedPrune: Neuron assignment ---
            m = neuron_pattern.search(line)
            if m:
                cid = m.group(1)
                cap = float(m.group(2))
                fedprune_neuron_data.append({
                    "Round": current_indices_round,
                    "client_id": cid,
                    "capacity": cap,
                    "neurons": int(m.group(3)),
                })
                # Backfill capacity from ratio if not registered
                if cid not in fedprune_capacity_data:
                    fedprune_capacity_data[cid] = cap
                continue

            # --- FedPrune: Extraction dynamics ---
            m = extraction_pattern.search(line)
            if m:
                fedprune_extraction_data.append({
                    "Round": current_indices_round,
                    "cap_std": float(m.group(1)),
                    "imp_frac": float(m.group(2)),
                })
                continue

            # --- Client: Neurons trained ---
            m = client_neurons_pattern.search(line)
            if m:
                client_neurons_buf[m.group(1)] = int(m.group(2))
                continue

            # --- Client: Training completion ---
            m = client_train_pattern.search(line)
            if m:
                client_train_buf[m.group(1)] = {
                    "samples": int(m.group(2)),
                    "time_s": float(m.group(3)),
                }
                continue

            # --- Client: Resource usage ---
            m = client_resource_pattern.search(line)
            if m:
                client_resource_buf[m.group(1)] = {
                    "gpu_mb": float(m.group(2)),
                    "ram_mb": float(m.group(3)),
                }
                continue

            # --- Client: Profile ---
            m = device_profile_pattern.search(line)
            if m:
                client_profiles[m.group(1)] = m.group(2)
                continue
            m = network_profile_pattern.search(line)
            if m:
                client_profiles[m.group(1)] = m.group(2)
                continue

    # Build DataFrames
    df_eval = pd.DataFrame(rounds_data)
    df_prof = pd.DataFrame(profiling_data)

    df_neurons = pd.DataFrame(fedprune_neuron_data) if fedprune_neuron_data else pd.DataFrame()
    df_extraction = pd.DataFrame(fedprune_extraction_data) if fedprune_extraction_data else pd.DataFrame()
    df_client_train = pd.DataFrame(fedprune_client_training) if fedprune_client_training else pd.DataFrame()

    is_fedprune = len(fedprune_neuron_data) > 0

    fedprune_data = {
        "capacities": fedprune_capacity_data,
        "profiles": client_profiles,
        "df_neurons": df_neurons,
        "df_extraction": df_extraction,
        "df_client_train": df_client_train,
        "is_fedprune": is_fedprune,
    }

    if is_fedprune:
        n_clients = len(fedprune_capacity_data) or (
            df_neurons["client_id"].nunique() if not df_neurons.empty else 0)
        print(f"   FedPrune detected: {n_clients} clients, "
              f"{len(df_neurons)} neuron assignments, "
              f"{len(df_client_train)} training records")

    return df_eval, df_prof, fedprune_data


def _flush_client_round(accumulator, round_num, capacities,
                         neurons_dict, train_dict, resource_dict):
    """Merge per-client data from one round into the accumulator list."""
    all_clients = set(neurons_dict) | set(train_dict) | set(resource_dict)
    for cid in all_clients:
        accumulator.append({
            "Round": round_num,
            "client_id": cid,
            "capacity": capacities.get(cid, None),
            "neurons": neurons_dict.get(cid, None),
            "samples": train_dict.get(cid, {}).get("samples", None),
            "time_s": train_dict.get(cid, {}).get("time_s", None),
            "gpu_mb": resource_dict.get(cid, {}).get("gpu_mb", None),
            "ram_mb": resource_dict.get(cid, {}).get("ram_mb", None),
        })


# =============================================================================
# PDF Report Generator
# =============================================================================

CLIENT_COLORS = list(mcolors.TABLEAU_COLORS.values())


def _client_color(cid, sorted_clients):
    idx = sorted_clients.index(cid) if cid in sorted_clients else 0
    return CLIENT_COLORS[idx % len(CLIENT_COLORS)]


def generate_pdf_report(df_eval, df_prof, fedprune_data, log_filename,
                         output_dir="../analysis"):
    """Generates a multi-page PDF report with plots."""

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(log_filename).replace(".log", "")
    pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")

    print(f"📊 Generating PDF report: {pdf_path}")

    is_fp = fedprune_data.get("is_fedprune", False)

    with PdfPages(pdf_path) as pdf:

        # =============================================================
        # PAGE 1: Title and Summary
        # =============================================================
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()

        strategy = "FedPrune" if is_fp else "Federated Learning"
        title = f"{strategy} Simulation Report\n{base_name}"

        lines = [
            f"Total Rounds Completed: {len(df_eval)}",
            f"Final Global Accuracy: {df_eval['Accuracy'].iloc[-1]:.2f}%"
            f" (Round {df_eval['Round'].iloc[-1]})",
            f"Final Global Loss: {df_eval['Loss'].iloc[-1]:.4f}",
            "",
            f"Average Aggregation Time: {df_prof['Aggregation_Time_s'].mean():.2f} seconds",
            f"Peak Server RAM Usage: {df_prof['RAM_Usage_MB'].max():.2f} MB",
            f"Peak Server CPU Usage: {df_prof['CPU_Usage_%'].max():.2f}%",
        ]

        if is_fp:
            caps = fedprune_data["capacities"]
            profs = fedprune_data["profiles"]
            df_ext = fedprune_data["df_extraction"]
            lines += ["", "--- FedPrune Configuration ---",
                       f"Clients: {len(caps)}"]
            for cid in sorted(caps.keys()):
                p = profs.get(cid, "n/a")
                lines.append(f"  {cid}: capacity={caps[cid]:.2f}  profile={p}")
            if not df_ext.empty:
                lines.append(f"Avg imp_frac: {df_ext['imp_frac'].mean():.3f}")
                lines.append(f"Avg cap_std:  {df_ext['cap_std'].mean():.3f}")

        fig_text.text(0.5, 0.90, title, transform=fig_text.transFigure,
                      size=18, ha="center", weight='bold')
        fig_text.text(0.08, 0.35, "\n".join(lines),
                      transform=fig_text.transFigure, size=11, ha="left",
                      family='monospace', va='bottom')
        pdf.savefig(fig_text)
        plt.close()

        # =============================================================
        # PAGE 2: Convergence (Accuracy & Loss)
        # =============================================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
        fig.suptitle("Global Model Convergence", fontsize=16, weight='bold')

        ax1.plot(df_eval['Round'], df_eval['Accuracy'],
                 marker='o', color='blue', linewidth=2)
        ax1.set_title("Global Test Accuracy over Rounds")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2.plot(df_eval['Round'], df_eval['Loss'],
                 marker='x', color='red', linewidth=2)
        ax2.set_title("Global Test Loss over Rounds")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle='--', alpha=0.7)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # =============================================================
        # PAGE 3: Server Profiling (Time & Memory)
        # =============================================================
        if not df_prof.empty:
            fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(8.5, 11))
            fig.suptitle("Server Resource Profiling", fontsize=16, weight='bold')

            ax3.plot(df_prof['Round'], df_prof['Aggregation_Time_s'],
                     marker='s', color='green', linewidth=2)
            ax3.set_title("Time Spent Aggregating per Round")
            ax3.set_xlabel("Communication Round")
            ax3.set_ylabel("Time (seconds)")
            ax3.grid(True, linestyle='--', alpha=0.7)

            ax4.plot(df_prof['Round'], df_prof['RAM_Usage_MB'],
                     marker='d', color='purple', linewidth=2)
            ax4.set_title("Server RAM Usage per Round")
            ax4.set_xlabel("Communication Round")
            ax4.set_ylabel("Memory (MB)")
            ax4.grid(True, linestyle='--', alpha=0.7)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

        # =============================================================
        # FEDPRUNE PAGES (auto-detected, skipped for other strategies)
        # =============================================================
        if not is_fp:
            print("✅ Report generated successfully.")
            return

        df_neurons = fedprune_data["df_neurons"]
        df_extraction = fedprune_data["df_extraction"]
        df_ct = fedprune_data["df_client_train"]
        capacities = fedprune_data["capacities"]
        profiles = fedprune_data["profiles"]
        all_clients = sorted(capacities.keys())

        # =============================================================
        # PAGE 4: Neuron Allocation per Client
        # =============================================================
        if not df_neurons.empty:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 11))
            fig.suptitle("FedPrune: Neuron Allocation",
                         fontsize=16, weight='bold')

            # Top: neurons per client over rounds
            for cid in all_clients:
                cd = df_neurons[df_neurons["client_id"] == cid]
                if cd.empty:
                    continue
                col = _client_color(cid, all_clients)
                cap = capacities.get(cid, 0.5)
                prof = profiles.get(cid, "?")
                ax_top.plot(cd["Round"], cd["neurons"], marker='o',
                           color=col, linewidth=2, markersize=4,
                           label=f"{cid} ({prof}, {cap:.1f})")

            ax_top.set_title("Neurons Assigned per Client over Rounds")
            ax_top.set_xlabel("Communication Round")
            ax_top.set_ylabel("Neurons Assigned")
            ax_top.legend(fontsize=8, loc='center left',
                         bbox_to_anchor=(1.01, 0.5))
            ax_top.grid(True, linestyle='--', alpha=0.7)

            # Bottom: average neurons bar chart
            avg = df_neurons.groupby("client_id")["neurons"].mean()
            avg = avg.reindex(all_clients)
            colors = [_client_color(c, all_clients) for c in all_clients]

            ax_bot.bar(range(len(all_clients)), avg.values,
                      color=colors, edgecolor='black', linewidth=0.5)
            for i, (cid, val) in enumerate(zip(all_clients, avg.values)):
                cap = capacities.get(cid, 0.5)
                ax_bot.text(i, val + 2, f"cap={cap:.1f}",
                           ha='center', fontsize=8, fontweight='bold')

            ax_bot.set_title("Average Neurons Assigned per Client")
            ax_bot.set_xlabel("Client")
            ax_bot.set_ylabel("Average Neurons")
            ax_bot.set_xticks(range(len(all_clients)))
            ax_bot.set_xticklabels(
                [f"{c}\n({profiles.get(c, '?')})" for c in all_clients],
                fontsize=8)
            ax_bot.grid(True, linestyle='--', alpha=0.7, axis='y')

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

        # =============================================================
        # PAGE 5: Extraction Dynamics (imp_frac + cap_std)
        # =============================================================
        if not df_extraction.empty:
            df_ed = df_extraction.drop_duplicates(
                subset="Round", keep="first")

            fig, (ax_e1, ax_e2) = plt.subplots(2, 1, figsize=(8.5, 11))
            fig.suptitle("FedPrune: Extraction Dynamics",
                         fontsize=16, weight='bold')

            ax_e1.plot(df_ed["Round"], df_ed["imp_frac"],
                      marker='s', color='#d62728', linewidth=2)
            ax_e1.axhline(y=0.5, color='gray', linestyle=':',
                         alpha=0.7, label="Min (uniform caps)")
            ax_e1.axhline(y=0.9, color='gray', linestyle=':',
                         alpha=0.7, label="Max (diverse caps)")
            ax_e1.fill_between(df_ed["Round"], 0.5, df_ed["imp_frac"],
                              alpha=0.15, color='#d62728')
            ax_e1.set_title("Importance Fraction (imp_frac) over Rounds")
            ax_e1.set_xlabel("Communication Round")
            ax_e1.set_ylabel("imp_frac")
            ax_e1.set_ylim(0.4, 1.0)
            ax_e1.legend(fontsize=9)
            ax_e1.grid(True, linestyle='--', alpha=0.7)

            ax_e2.plot(df_ed["Round"], df_ed["cap_std"],
                      marker='d', color='#1f77b4', linewidth=2)
            ax_e2.fill_between(df_ed["Round"], 0, df_ed["cap_std"],
                              alpha=0.15, color='#1f77b4')
            ax_e2.set_title("Capacity Standard Deviation over Rounds")
            ax_e2.set_xlabel("Communication Round")
            ax_e2.set_ylabel("std(capacities)")
            ax_e2.grid(True, linestyle='--', alpha=0.7)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

        # =============================================================
        # PAGE 6: Per-Client Training Time
        # =============================================================
        if not df_ct.empty and "time_s" in df_ct.columns:
            df_t = df_ct.dropna(subset=["time_s"])

            if not df_t.empty:
                fig, (ax_t1, ax_t2) = plt.subplots(2, 1, figsize=(8.5, 11))
                fig.suptitle("FedPrune: Per-Client Training Efficiency",
                             fontsize=16, weight='bold')

                for cid in all_clients:
                    cd = df_t[df_t["client_id"] == cid]
                    if cd.empty:
                        continue
                    col = _client_color(cid, all_clients)
                    cap = capacities.get(cid, 0.5)
                    prof = profiles.get(cid, "?")
                    ax_t1.plot(cd["Round"], cd["time_s"], marker='o',
                             color=col, linewidth=2, markersize=4,
                             label=f"{cid} ({prof}, {cap:.1f})")

                ax_t1.set_title("Local Training Time per Client per Round")
                ax_t1.set_xlabel("Communication Round")
                ax_t1.set_ylabel("Training Time (seconds)")
                ax_t1.legend(fontsize=8, loc='center left',
                            bbox_to_anchor=(1.01, 0.5))
                ax_t1.grid(True, linestyle='--', alpha=0.7)

                # Bar chart: average time
                avg_t = df_t.groupby("client_id")["time_s"].mean()
                avg_t = avg_t.reindex(all_clients)
                colors = [_client_color(c, all_clients) for c in all_clients]

                ax_t2.bar(range(len(all_clients)), avg_t.values,
                         color=colors, edgecolor='black', linewidth=0.5)
                for i, (cid, val) in enumerate(zip(all_clients, avg_t.values)):
                    if val is not None and not np.isnan(val):
                        ax_t2.text(i, val + 0.05, f"{val:.2f}s",
                                  ha='center', fontsize=8, fontweight='bold')

                ax_t2.set_title("Average Training Time per Client")
                ax_t2.set_xlabel("Client")
                ax_t2.set_ylabel("Avg Training Time (s)")
                ax_t2.set_xticks(range(len(all_clients)))
                ax_t2.set_xticklabels(
                    [f"{c}\n({profiles.get(c, '?')})" for c in all_clients],
                    fontsize=8)
                ax_t2.grid(True, linestyle='--', alpha=0.7, axis='y')

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close()

        # =============================================================
        # PAGE 7: Per-Client Resource Usage (GPU + RAM)
        # =============================================================
        if not df_ct.empty and "gpu_mb" in df_ct.columns:
            df_r = df_ct.dropna(subset=["gpu_mb", "ram_mb"])

            if not df_r.empty:
                fig, (ax_r1, ax_r2) = plt.subplots(2, 1, figsize=(8.5, 11))
                fig.suptitle("FedPrune: Per-Client Resource Usage",
                             fontsize=16, weight='bold')

                for cid in all_clients:
                    cd = df_r[df_r["client_id"] == cid]
                    if cd.empty:
                        continue
                    col = _client_color(cid, all_clients)
                    prof = profiles.get(cid, "?")
                    ax_r1.plot(cd["Round"], cd["gpu_mb"], marker='o',
                             color=col, linewidth=1.5, markersize=3,
                             label=f"{cid} ({prof})")

                ax_r1.set_title("Peak GPU Memory per Client per Round")
                ax_r1.set_xlabel("Communication Round")
                ax_r1.set_ylabel("GPU Memory (MB)")
                ax_r1.legend(fontsize=8, loc='center left',
                            bbox_to_anchor=(1.01, 0.5))
                ax_r1.grid(True, linestyle='--', alpha=0.7)

                for cid in all_clients:
                    cd = df_r[df_r["client_id"] == cid]
                    if cd.empty:
                        continue
                    col = _client_color(cid, all_clients)
                    ax_r2.plot(cd["Round"], cd["ram_mb"], marker='o',
                             color=col, linewidth=1.5, markersize=3,
                             label=f"{cid}")

                ax_r2.set_title("Client RAM Usage per Round")
                ax_r2.set_xlabel("Communication Round")
                ax_r2.set_ylabel("RAM (MB)")
                ax_r2.legend(fontsize=8, loc='center left',
                            bbox_to_anchor=(1.01, 0.5))
                ax_r2.grid(True, linestyle='--', alpha=0.7)

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close()

    print("✅ Report generated successfully.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = find_latest_log(log_dir="../fl_logs")

    df_eval, df_prof, fedprune_data = parse_fl_log(log_file)

    if df_eval.empty:
        print("⚠️ No evaluation metrics found. Did the simulation complete?")
    else:
        generate_pdf_report(df_eval, df_prof, fedprune_data, log_file,
                           output_dir=".")