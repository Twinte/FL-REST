#!/bin/bash
set -e # Exit immediately if any command fails

#################################################################
# 1. SIMULATION PARAMETERS
#################################################################

# --- Client Profiles ---
CLIENTS_NORMAL=2
CLIENTS_SLOW=2
CLIENTS_LOSSY=1

# Derived Totals (Exported for Python)
export TOTAL_CLIENTS=$(($CLIENTS_NORMAL + $CLIENTS_SLOW + $CLIENTS_LOSSY))
export MIN_CLIENTS_PER_ROUND=$TOTAL_CLIENTS
export MIN_CLIENTS_FOR_AGGREGATION=3

export TOTAL_ROUNDS=10

# --- Client Training Config ---
export LOCAL_EPOCHS=3
export BATCH_SIZE=32
export LEARNING_RATE=0.01
export MOMENTUM=0.9

# --- Robustness & Strategy Config ---
export CLIENT_ALGO="FedProx"
export FEDPROX_MU=0.01
export AGGREGATION_STRATEGY="FedAvgM"
export SERVER_LEARNING_RATE=1.0
export SERVER_MOMENTUM=0.9

# --- Data & Sim Config ---
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
export MODEL_NAME="SimpleCNN"

#################################################################
# 2. GENERATE DOCKER-COMPOSE
#################################################################

echo "▶️ Starting simulation with $TOTAL_CLIENTS clients..."
echo "🔄 Generating docker-compose.yml..."

# Pass client counts to the generator
python generate_compose.py \
  --normal $CLIENTS_NORMAL \
  --slow $CLIENTS_SLOW \
  --lossy $CLIENTS_LOSSY

echo "✅ docker-compose.yml generated."

#################################################################
# 3. EXECUTE SIMULATION
#################################################################

echo "🧹 Cleaning up old containers..."
docker compose down --remove-orphans

mkdir -p fl_logs
LOG_FILE="fl_logs/simulation_$(date +'%Y%m%d_%H%M%S').log"

echo "🚀 Building images..."
docker compose build

echo "📦 Preparing Data..."
# We run this just to download data/create partitions. 
# It inherits env vars automatically.
docker compose run --rm server python prepare_data.py

echo "▶️ Starting Simulation..."
echo "🪵 Log file: $LOG_FILE"

# The env vars exported above are automatically passed to Docker Compose
docker compose up --remove-orphans --exit-code-from server 2>&1 | tee $LOG_FILE

echo "---"
echo "✅ Simulation complete."