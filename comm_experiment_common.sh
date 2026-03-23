#!/bin/bash
# =================================================================
# comm_experiment_common.sh
# =================================================================
# Shared configuration for all communication experiments.
# Sourced by the individual experiment scripts — not run directly.
# =================================================================

# --- Cleanup handler ---
cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] Caught interrupt. Stopping Docker containers..."
    docker compose down --remove-orphans 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# --- Fixed parameters across ALL communication experiments ---
export LOCAL_EPOCHS=3
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export MOMENTUM=0.9
export SERVER_LEARNING_RATE=1.0
export SERVER_MOMENTUM=0.9
export EMA_DECAY=0.9
export IMPORTANCE_ALPHA=0.3
export MODEL_NAME="FedPruneCNN100"
export DATASET_NAME="CIFAR100"
export CLIENT_DROPOUT_RATE=0.0
export SLOW_SENDER_RATE=0.0
export SLOW_SENDER_DELAY_SEC=30
export NETWORK_LATENCY_RATE=0.0
export NETWORK_LATENCY_DELAY_SEC=5
export DEVICE="auto"
export POLL_INTERVAL=1          # Fast polling for accurate timing
export SAVED_MODEL_NAME="final_global_model.pth"
export FEDPROX_MU=0.01

# --- Helper function: run one experiment ---
run_single_experiment() {
    local METHOD=$1
    local SCENARIO=$2
    local ROUNDS=$3
    local BANDWIDTH=$4    # "none", "1mbps", "5mbps", "10mbps"
    local SEED=$5
    local TAG=$6          # e.g., "A1", "B4", "C2"

    # --- Method-specific config ---
    export CLIENT_ALGO="$METHOD"
    export AGGREGATION_STRATEGY="$METHOD"
    export RANDOM_SEED=$SEED
    export TOTAL_ROUNDS=$ROUNDS

    # --- Scenario-specific config ---
    case $SCENARIO in
        stress_test)
            CLIENTS_HIGH_PERF=1
            CLIENTS_MID_PERF=2
            CLIENTS_LOW_PERF=2
            export DIRICHLET_ALPHA=0.1
            export ROUND_TIMEOUT_SEC=120
            ;;
        mixed_capacities)
            CLIENTS_HIGH_PERF=1
            CLIENTS_MID_PERF=2
            CLIENTS_LOW_PERF=2
            export DIRICHLET_ALPHA=0.5
            export ROUND_TIMEOUT_SEC=120
            ;;
        high_heterogeneity)
            CLIENTS_HIGH_PERF=1
            CLIENTS_MID_PERF=2
            CLIENTS_LOW_PERF=2
            export DIRICHLET_ALPHA=0.1
            export ROUND_TIMEOUT_SEC=120
            ;;
        standard)
            CLIENTS_HIGH_PERF=1
            CLIENTS_MID_PERF=2
            CLIENTS_LOW_PERF=2
            export DIRICHLET_ALPHA=0.5
            export ROUND_TIMEOUT_SEC=120
            ;;
    esac

    export TOTAL_CLIENTS=$(($CLIENTS_HIGH_PERF + $CLIENTS_MID_PERF + $CLIENTS_LOW_PERF))
    export MIN_CLIENTS_PER_ROUND=$TOTAL_CLIENTS
    export MIN_CLIENTS_FOR_AGGREGATION=$TOTAL_CLIENTS

    # --- Compose generation with optional bandwidth cap ---
    local COMPOSE_ARGS="--high_perf $CLIENTS_HIGH_PERF --mid_perf $CLIENTS_MID_PERF --low_perf $CLIENTS_LOW_PERF"
    if [ "$BANDWIDTH" != "none" ]; then
        COMPOSE_ARGS="$COMPOSE_ARGS --bandwidth $BANDWIDTH"
    fi

    # --- Log naming ---
    local LOG_DIR="fl_logs/comm_experiments"
    mkdir -p "$LOG_DIR"
    local LOG_FILE="${LOG_DIR}/${TAG}_${METHOD}_${SCENARIO}_bw${BANDWIDTH}_seed${SEED}.log"

    # --- Clear round_metrics.jsonl for this run ---
    rm -f fl_logs/round_metrics.jsonl

    echo ""
    echo "================================================================"
    echo "  [$TAG] $METHOD | $SCENARIO | bw=$BANDWIDTH | seed=$SEED"
    echo "  Rounds: $ROUNDS | Clients: $TOTAL_CLIENTS"
    echo "  Log: $LOG_FILE"
    echo "================================================================"

    # --- Generate compose, build, run ---
    python generate_compose.py $COMPOSE_ARGS
    docker compose down --remove-orphans 2>/dev/null || true
    docker compose build --quiet
    docker compose run --rm server python prepare_data.py 2>/dev/null
    docker compose up --remove-orphans --exit-code-from server 2>&1 | tee "$LOG_FILE"
    docker compose down --remove-orphans 2>/dev/null || true

    # --- Copy round_metrics.jsonl with unique name ---
    if [ -f "fl_logs/round_metrics.jsonl" ]; then
        cp fl_logs/round_metrics.jsonl "${LOG_DIR}/${TAG}_${METHOD}_${SCENARIO}_bw${BANDWIDTH}_seed${SEED}_metrics.jsonl"
    fi

    echo "  [$TAG] Complete."
}
