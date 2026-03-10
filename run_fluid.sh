#!/bin/bash
set -e

# Clean shutdown on Ctrl+C or kill
cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] Caught interrupt. Stopping Docker containers..."
    docker compose down --remove-orphans 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

#################################################################
# FLuID Baseline (Yang et al., AAAI 2023)
# Leader-client invariant dropout
# 
# NOTE: FLuID requires at least 1 "leader" client with capacity
# >= 0.6 (threshold). Leaders train the full model; their update
# deltas drive importance for straggler submodel extraction.
#################################################################

CLIENTS_HIGH_PERF=1
CLIENTS_MID_PERF=2
CLIENTS_LOW_PERF=2

export TOTAL_CLIENTS=$(($CLIENTS_HIGH_PERF + $CLIENTS_MID_PERF + $CLIENTS_LOW_PERF))
export MIN_CLIENTS_PER_ROUND=$TOTAL_CLIENTS
export MIN_CLIENTS_FOR_AGGREGATION=3
export TOTAL_ROUNDS=100

export LOCAL_EPOCHS=3
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export MOMENTUM=0.9

# --- Strategy: FLuID ---
export CLIENT_ALGO="FLuID"
export AGGREGATION_STRATEGY="FLuID"
export SERVER_LEARNING_RATE=1.0
export SERVER_MOMENTUM=0.9
export EMA_DECAY=0.9
export IMPORTANCE_ALPHA=0.3

export MODEL_NAME="FedPruneCNN"

export DATASET_NAME="CIFAR10"
export DIRICHLET_ALPHA=0.5
export CLIENT_DROPOUT_RATE=0.0
export ROUND_TIMEOUT_SEC=300
export SLOW_SENDER_RATE=0.0
export SLOW_SENDER_DELAY_SEC=30
export NETWORK_LATENCY_RATE=0.0
export NETWORK_LATENCY_DELAY_SEC=5
export DEVICE="auto"
export POLL_INTERVAL=10
export RANDOM_SEED=42
export SAVED_MODEL_NAME="final_global_model.pth"
export FEDPROX_MU=0.01

echo "Starting FLuID baseline with $TOTAL_CLIENTS clients..."
echo "   Model: $MODEL_NAME | Strategy: $AGGREGATION_STRATEGY"
echo "   Leaders: high_perf clients (capacity >= 0.6)"

python generate_compose.py \
  --high_perf $CLIENTS_HIGH_PERF \
  --mid_perf $CLIENTS_MID_PERF \
  --low_perf $CLIENTS_LOW_PERF

docker compose down --remove-orphans
mkdir -p fl_logs
LOG_FILE="fl_logs/simulation_$(date +'%Y%m%d_%H%M%S').log"

docker compose build
docker compose run --rm server python prepare_data.py

echo "Starting FLuID Simulation..."
docker compose up --remove-orphans --exit-code-from server 2>&1 | tee $LOG_FILE

echo "FLuID simulation complete."
