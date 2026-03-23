#!/bin/bash
set -e
# =================================================================
# Experiment C: Time-to-Accuracy Under Bandwidth Constraints
# =================================================================
# Purpose: Produce the key paper figure — accuracy vs wall-clock time.
#          When plotted by round, FedPrune ≈ FIARSE. But when plotted
#          by time, FIARSE rounds are 5-8× longer → FedPrune converges
#          dramatically faster in real-time.
#
# Bandwidth: 5 Mbps (realistic IoT/constrained wireless)
# Scenarios: stress_test (50 rounds) + high_heterogeneity (100 rounds)
# Methods:   FedPrune, FIARSE, FedRolex
#
# Expected: FedPrune reaches 60% accuracy while FIARSE is still
#           at 40-50% at the same wall-clock mark.
#
# Total runs: 6 (~1-2 hours each)
# =================================================================

source "$(dirname "$0")/comm_experiment_common.sh"

BANDWIDTH="1mbps"
SEED=42

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT C: Time-to-Accuracy at 1 Mbps           ║"
echo "║  The 'money figure' for the paper                   ║"
echo "╚══════════════════════════════════════════════════════╝"

# --- Stress Test (50 rounds) ---
echo ""
echo "--- Stress Test (50 rounds, α=0.1) ---"
run_single_experiment "FedPrune"  "stress_test" 50 $BANDWIDTH $SEED "C1"
run_single_experiment "FIARSE"   "stress_test" 50 $BANDWIDTH $SEED "C2"
run_single_experiment "FedRolex" "stress_test" 50 $BANDWIDTH $SEED "C3"

# --- High Heterogeneity (100 rounds) ---
echo ""
echo "--- High Heterogeneity (100 rounds, α=0.1) ---"
run_single_experiment "FedPrune"  "high_heterogeneity" 100 $BANDWIDTH $SEED "C4"
run_single_experiment "FIARSE"   "high_heterogeneity" 100 $BANDWIDTH $SEED "C5"
run_single_experiment "FedRolex" "high_heterogeneity" 100 $BANDWIDTH $SEED "C6"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT C COMPLETE                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Generate time-to-accuracy plots with:"
echo "  python plot_time_to_accuracy.py \\"
echo "    fl_logs/comm_experiments/C1_*_metrics.jsonl \\"
echo "    fl_logs/comm_experiments/C2_*_metrics.jsonl \\"
echo "    fl_logs/comm_experiments/C3_*_metrics.jsonl"
