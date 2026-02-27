import os
import re
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def find_latest_log(log_dir="../fl_logs"):
    """Finds the most recent simulation_*.log file."""
    search_pattern = os.path.join(log_dir, "simulation_*.log")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print(f"No log files found in {log_dir}")
        sys.exit(1)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def parse_fl_log(log_path):
    """
    Reads the unstructured Docker log and extracts key metrics using Regex.
    Returns Pandas DataFrames for easy plotting.
    """
    print(f"Parsing log file: {log_path}")
    
    rounds_data = []
    profiling_data = []
    
    # Regex Patterns based on server/app.py log formats
    # Matches: "Round 0 Complete. Accuracy: 45.20%, Loss: 1.5432"
    eval_pattern = re.compile(r"Round (\d+) Complete\. Accuracy: ([\d\.]+)%, Loss: ([\d\.]+)")
    
    # Matches: "PROFILING: CPU 15.2%, RAM 450.50MB, Time 2.34s"
    # Note: We tie this to the current round by tracking the last seen round
    prof_pattern = re.compile(r"PROFILING: CPU ([\d\.]+)%, RAM ([\d\.]+)MB, Time ([\d\.]+)s")
    
    current_round = 0
    
    with open(log_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check for Evaluation Metrics
            eval_match = eval_pattern.search(line)
            if eval_match:
                current_round = int(eval_match.group(1))
                accuracy = float(eval_match.group(2))
                loss = float(eval_match.group(3))
                rounds_data.append({
                    "Round": current_round,
                    "Accuracy": accuracy,
                    "Loss": loss
                })
                continue
            
            # Check for Hardware Profiling Metrics
            prof_match = prof_pattern.search(line)
            if prof_match:
                cpu = float(prof_match.group(1))
                ram = float(prof_match.group(2))
                agg_time = float(prof_match.group(3))
                profiling_data.append({
                    "Round": current_round, # Maps to the round that just evaluated
                    "CPU_Usage_%": cpu,
                    "RAM_Usage_MB": ram,
                    "Aggregation_Time_s": agg_time
                })

    df_eval = pd.DataFrame(rounds_data)
    df_prof = pd.DataFrame(profiling_data)
    
    return df_eval, df_prof

def generate_pdf_report(df_eval, df_prof, log_filename, output_dir="../analysis"):
    """Generates a multi-page PDF report with plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract timestamp from log filename to name the PDF
    base_name = os.path.basename(log_filename).replace(".log", "")
    pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
    
    print(f"Generating PDF report: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        
        # --- PAGE 1: Title and Summary Text ---
        fig_text = plt.figure(figsize=(8.5, 11))
        fig_text.clf()
        
        title = f"Federated Learning Simulation Report\n{base_name}"
        summary = (
            f"Total Rounds Completed: {len(df_eval)}\n"
            f"Final Global Accuracy: {df_eval['Accuracy'].iloc[-1]:.2f}% (Round {df_eval['Round'].iloc[-1]})\n"
            f"Final Global Loss: {df_eval['Loss'].iloc[-1]:.4f}\n\n"
            f"Average Aggregation Time: {df_prof['Aggregation_Time_s'].mean():.2f} seconds\n"
            f"Peak Server RAM Usage: {df_prof['RAM_Usage_MB'].max():.2f} MB\n"
            f"Peak Server CPU Usage: {df_prof['CPU_Usage_%'].max():.2f}%\n"
        )
        
        fig_text.text(0.5, 0.85, title, transform=fig_text.transFigure, size=18, ha="center", weight='bold')
        fig_text.text(0.1, 0.6, summary, transform=fig_text.transFigure, size=12, ha="left", family='monospace')
        pdf.savefig(fig_text)
        plt.close()

        # --- PAGE 2: Convergence Metrics (Accuracy & Loss) ---
        fig_metrics, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
        fig_metrics.suptitle("Global Model Convergence", fontsize=16, weight='bold')
        
        # Accuracy Plot
        ax1.plot(df_eval['Round'], df_eval['Accuracy'], marker='o', color='blue', linestyle='-', linewidth=2)
        ax1.set_title("Global Test Accuracy over Rounds")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Accuracy (%)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Loss Plot
        ax2.plot(df_eval['Round'], df_eval['Loss'], marker='x', color='red', linestyle='-', linewidth=2)
        ax2.set_title("Global Test Loss over Rounds")
        ax2.set_xlabel("Communication Round")
        ax2.set_ylabel("Loss")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        fig_metrics.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig_metrics)
        plt.close()

        # --- PAGE 3: Server Profiling (Time & Memory) ---
        if not df_prof.empty:
            fig_prof, (ax3, ax4) = plt.subplots(2, 1, figsize=(8.5, 11))
            fig_prof.suptitle("Server Resource Profiling", fontsize=16, weight='bold')
            
            # Aggregation Time
            ax3.plot(df_prof['Round'], df_prof['Aggregation_Time_s'], marker='s', color='green', linewidth=2)
            ax3.set_title("Time Spent Aggregating per Round")
            ax3.set_xlabel("Communication Round")
            ax3.set_ylabel("Time (seconds)")
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # RAM Usage
            ax4.plot(df_prof['Round'], df_prof['RAM_Usage_MB'], marker='d', color='purple', linewidth=2)
            ax4.set_title("Server RAM Usage per Round")
            ax4.set_xlabel("Communication Round")
            ax4.set_ylabel("Memory (MB)")
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            fig_prof.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig_prof)
            plt.close()

    print("Report generated successfully.")

if __name__ == "__main__":
    # If a specific log file is passed via CLI, use it. Otherwise, auto-detect latest.
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Assuming script is run from inside the 'analysis' directory
        log_file = find_latest_log(log_dir="../fl_logs")
        
    df_eval, df_prof = parse_fl_log(log_file)
    
    if df_eval.empty:
        print("No evaluation metrics found in the log. Did the simulation complete any rounds?")
    else:
        generate_pdf_report(df_eval, df_prof, log_file, output_dir=".")