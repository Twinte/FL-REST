#!/bin/bash
set -e
# =================================================================
# run_all_comm_experiments.sh
# =================================================================
# Runs all three experiment batches in sequence.
#
# Usage:
#   ./run_all_comm_experiments.sh          # Run everything
#   ./run_all_comm_experiments.sh A        # Only Experiment A
#   ./run_all_comm_experiments.sh B        # Only Experiment B
#   ./run_all_comm_experiments.sh C        # Only Experiment C
#   ./run_all_comm_experiments.sh A B      # Experiments A and B
#
# Total time estimate:
#   A: ~1 hour    (5 runs × ~12 min)
#   B: ~3 hours   (9 runs × ~20 min at low bandwidth)
#   C: ~8 hours   (6 runs × ~1-2 hours)
#   Total: ~12 hours
# =================================================================

SCRIPT_DIR="$(dirname "$0")"
EXPERIMENTS="${@:-A B C}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  FL-REST Communication Experiments                       ║"
echo "║  Running: $EXPERIMENTS                                   ║"
echo "║  Started: $(date)                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Build Docker image once (shared across all runs)
echo "Building Docker image..."
docker compose build --quiet 2>/dev/null || true

START_TIME=$SECONDS

for EXP in $EXPERIMENTS; do
    case $EXP in
        A|a) bash "$SCRIPT_DIR/run_experiment_A.sh" ;;
        B|b) bash "$SCRIPT_DIR/run_experiment_B.sh" ;;
        C|c) bash "$SCRIPT_DIR/run_experiment_C.sh" ;;
        *)   echo "Unknown experiment: $EXP (expected A, B, or C)" ;;
    esac
done

ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE                                ║"
echo "║  Total time: ${HOURS}h ${MINS}m                         ║"
echo "║  Results in: fl_logs/comm_experiments/                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. python parse_comm_metrics.py --compare fl_logs/comm_experiments/A*.log"
echo "  2. python parse_comm_metrics.py --compare fl_logs/comm_experiments/B*_bw5mbps_*.log"
echo "  3. python plot_time_to_accuracy.py fl_logs/comm_experiments/C*_metrics.jsonl"
