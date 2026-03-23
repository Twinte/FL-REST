#!/bin/bash
set -e
# =================================================================
# Experiment A: Payload Size Validation
# =================================================================
# Purpose: Confirm that submodel extraction produces different-sized
#          payloads per method. No bandwidth cap — pure size measurement.
#
# Scenario: mixed_capacities (diverse client caps → visible size differences)
# Rounds:   20 (enough to see steady-state payload sizes)
# Methods:  FedPrune, FIARSE, FedRolex, HeteroFL, FLuID
#
# Expected results:
#   FedPrune DL:  ~25-49% of full model (depends on client capacity)
#   FIARSE DL:    ~100% (full model required for importance computation)
#   FedRolex DL:  ~25-49% (same submodel sizes as FedPrune)
#   HeteroFL DL:  ~25-49% (same submodel sizes)
#   FLuID DL:     100% for leaders, ~25-49% for stragglers
#
# Total runs: 5 (~10-15 min each)
# =================================================================

source "$(dirname "$0")/comm_experiment_common.sh"

SCENARIO="mixed_capacities"
ROUNDS=20
BANDWIDTH="none"
SEED=42

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT A: Payload Size Validation               ║"
echo "║  Scenario: $SCENARIO | Rounds: $ROUNDS | No BW cap  ║"
echo "╚══════════════════════════════════════════════════════╝"

run_single_experiment "FedPrune"  "$SCENARIO" $ROUNDS $BANDWIDTH $SEED "A1"
run_single_experiment "FIARSE"   "$SCENARIO" $ROUNDS $BANDWIDTH $SEED "A2"
run_single_experiment "FedRolex" "$SCENARIO" $ROUNDS $BANDWIDTH $SEED "A3"
run_single_experiment "HeteroFL" "$SCENARIO" $ROUNDS $BANDWIDTH $SEED "A4"
run_single_experiment "FLuID"    "$SCENARIO" $ROUNDS $BANDWIDTH $SEED "A5"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  EXPERIMENT A COMPLETE                               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Analyze results with:"
echo "  python parse_comm_metrics.py --compare \\"
echo "    fl_logs/comm_experiments/A1_FedPrune_*.log \\"
echo "    fl_logs/comm_experiments/A2_FIARSE_*.log \\"
echo "    fl_logs/comm_experiments/A3_FedRolex_*.log \\"
echo "    fl_logs/comm_experiments/A4_HeteroFL_*.log \\"
echo "    fl_logs/comm_experiments/A5_FLuID_*.log"
