#!/bin/bash
set -e
# =================================================================
# Experiment B: Round Latency Under Bandwidth Constraints
# =================================================================
# Purpose: Measure actual round latency when bandwidth is capped.
#          FedPrune sends smaller payloads → faster rounds.
#          FIARSE sends full model → bottlenecked at low bandwidth.
#
# Scenario: stress_test (weakest clients, largest payload gap)
# Rounds:   20 (measure steady-state round latency)
# Methods:  FedPrune, FIARSE, FedRolex
# Bandwidth: 1 Mbps, 5 Mbps, 10 Mbps
#
# Expected results at 1 Mbps:
#   FedPrune round: ~3s   (small submodel transfers)
#   FIARSE round:   ~24s  (full model download to every client)
#   FedRolex round: ~3s   (same submodel size as FedPrune)
#
# Total runs: 9 (~15-20 min each at low bandwidth)
# =================================================================

source "$(dirname "$0")/comm_experiment_common.sh"

SCENARIO="stress_test"
ROUNDS=20
SEED=42

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT B: Round Latency vs Bandwidth            ║"
echo "║  Scenario: $SCENARIO | Rounds: $ROUNDS              ║"
echo "║  Bandwidth: 1 / 5 / 10 Mbps × 3 methods            ║"
echo "╚══════════════════════════════════════════════════════╝"

# --- 1 Mbps ---
run_single_experiment "FedPrune"  "$SCENARIO" $ROUNDS "1mbps"  $SEED "B1"
run_single_experiment "FIARSE"   "$SCENARIO" $ROUNDS "1mbps"  $SEED "B2"
run_single_experiment "FedRolex" "$SCENARIO" $ROUNDS "1mbps"  $SEED "B3"

# --- 5 Mbps ---
run_single_experiment "FedPrune"  "$SCENARIO" $ROUNDS "5mbps"  $SEED "B4"
run_single_experiment "FIARSE"   "$SCENARIO" $ROUNDS "5mbps"  $SEED "B5"
run_single_experiment "FedRolex" "$SCENARIO" $ROUNDS "5mbps"  $SEED "B6"

# --- 10 Mbps ---
run_single_experiment "FedPrune"  "$SCENARIO" $ROUNDS "10mbps" $SEED "B7"
run_single_experiment "FIARSE"   "$SCENARIO" $ROUNDS "10mbps" $SEED "B8"
run_single_experiment "FedRolex" "$SCENARIO" $ROUNDS "10mbps" $SEED "B9"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT B COMPLETE                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Analyze results with:"
echo "  # Per-bandwidth comparison:"
echo "  python parse_comm_metrics.py --compare \\"
echo "    fl_logs/comm_experiments/B1_FedPrune_*_bw1mbps_*.log \\"
echo "    fl_logs/comm_experiments/B2_FIARSE_*_bw1mbps_*.log \\"
echo "    fl_logs/comm_experiments/B3_FedRolex_*_bw1mbps_*.log"
echo ""
echo "  python parse_comm_metrics.py --compare \\"
echo "    fl_logs/comm_experiments/B4_FedPrune_*_bw5mbps_*.log \\"
echo "    fl_logs/comm_experiments/B5_FIARSE_*_bw5mbps_*.log \\"
echo "    fl_logs/comm_experiments/B6_FedRolex_*_bw5mbps_*.log"
