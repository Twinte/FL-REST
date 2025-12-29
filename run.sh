#!/bin/bash
set -e # Exit immediately if any command fails

#################################################################
# 1. SIMULATION PARAMETERS
# (Edit these values to configure your test run)
#################################################################

# --- Client Profiles ---
CLIENTS_NORMAL=2   # GPU + Fast Network
CLIENTS_SLOW=2     # CPU + 5mbit Bandwidth
CLIENTS_LOSSY=1    # GPU + 5% Packet Loss

# Calculate Totals
TOTAL_CLIENTS=$(($CLIENTS_NORMAL + $CLIENTS_SLOW + $CLIENTS_LOSSY))
MIN_CLIENTS_FOR_AGGREGATION=3

TOTAL_ROUNDS=10


# --- Client Training Config ---
LOCAL_EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=0.01
MOMENTUM=0.9

# --- NEW: Robustness & Strategy Config ---
# Client Side Algorithm: "Standard", "FedProx", "Scaffold"
CLIENT_ALGO="FedProx"
FEDPROX_MU=0.01  # Penalty weight for FedProx (0.01 - 1.0)

# Server Side Strategy: "FedAvg", "FedAvgM", "FedAdam", "Scaffold"
AGGREGATION_STRATEGY="FedAvgM"
SERVER_LEARNING_RATE=1.0  # How fast the global model updates (1.0 for FedAvg)
SERVER_MOMENTUM=0.9       # Server-side momentum (0.0 to 0.9)

# --- Data Config (Non-IID) ---
DIRICHLET_ALPHA=0.5

# --- FL Condition Simulation ---
CLIENT_DROPOUT_RATE=0.0
ROUND_TIMEOUT_SEC=300

# --- Network Traffic Simulation (Application Level) ---
SLOW_SENDER_RATE=0.0
SLOW_SENDER_DELAY_SEC=30
NETWORK_LATENCY_RATE=0.0
NETWORK_LATENCY_DELAY_SEC=5

# --- System ---
DEVICE="auto"
POLL_INTERVAL=10
RANDOM_SEED=42
SAVED_MODEL_NAME="final_global_model.pth"

# ---------------------------------------------------------------
# (Derived values - DO NOT EDIT)
# We must sum up the NEW variables we defined at the top
NUM_CLIENTS=$(($CLIENTS_NORMAL + $CLIENTS_SLOW + $CLIENTS_LOSSY))
MIN_CLIENTS_PER_ROUND=$NUM_CLIENTS
TOTAL_CLIENTS=$NUM_CLIENTS
CONFIG_FILE="config.py"
# ---------------------------------------------------------------


#################################################################
# 2. CONFIGURATION SCRIPT
# (This section automatically updates config.py)
#################################################################

echo "▶️ Starting simulation with $NUM_CLIENTS clients for $TOTAL_ROUNDS rounds..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found. Cannot configure run."
    exit 1
fi

echo "🔄 Updating $CONFIG_FILE with new parameters..."

# Basic FL Config
sed -i "s/^TOTAL_ROUNDS = .*/TOTAL_ROUNDS = $TOTAL_ROUNDS/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_PER_ROUND = .*/MIN_CLIENTS_PER_ROUND = $MIN_CLIENTS_PER_ROUND/" $CONFIG_FILE
sed -i "s/^TOTAL_CLIENTS = .*/TOTAL_CLIENTS = $TOTAL_CLIENTS/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_FOR_AGGREGATION = .*/MIN_CLIENTS_FOR_AGGREGATION = $MIN_CLIENTS_FOR_AGGREGATION/" $CONFIG_FILE

# Training Hyperparameters
sed -i "s/^LOCAL_EPOCHS = .*/LOCAL_EPOCHS = $LOCAL_EPOCHS/" $CONFIG_FILE
sed -i "s/^BATCH_SIZE = .*/BATCH_SIZE = $BATCH_SIZE/" $CONFIG_FILE
sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = $LEARNING_RATE/" $CONFIG_FILE
sed -i "s/^MOMENTUM = .*/MOMENTUM = $MOMENTUM/" $CONFIG_FILE

# --- NEW: Strategy & Algorithm Injection ---
sed -i "s/^CLIENT_ALGO = .*/CLIENT_ALGO = \"$CLIENT_ALGO\"/" $CONFIG_FILE
sed -i "s/^FEDPROX_MU = .*/FEDPROX_MU = $FEDPROX_MU/" $CONFIG_FILE
sed -i "s/^AGGREGATION_STRATEGY = .*/AGGREGATION_STRATEGY = \"$AGGREGATION_STRATEGY\"/" $CONFIG_FILE
sed -i "s/^SERVER_LEARNING_RATE = .*/SERVER_LEARNING_RATE = $SERVER_LEARNING_RATE/" $CONFIG_FILE
sed -i "s/^SERVER_MOMENTUM = .*/SERVER_MOMENTUM = $SERVER_MOMENTUM/" $CONFIG_FILE
# -------------------------------------------

# System & Simulation
sed -i "s/^POLL_INTERVAL = .*/POLL_INTERVAL = $POLL_INTERVAL/" $CONFIG_FILE
sed -i "s/^DIRICHLET_ALPHA = .*/DIRICHLET_ALPHA = $DIRICHLET_ALPHA/" $CONFIG_FILE
sed -i "s/^RANDOM_SEED = .*/RANDOM_SEED = $RANDOM_SEED/" $CONFIG_FILE
sed -i "s/^SAVED_MODEL_NAME = .*/SAVED_MODEL_NAME = \"$SAVED_MODEL_NAME\"/" $CONFIG_FILE
sed -i "s/^DEVICE = .*/DEVICE = \"$DEVICE\"/" $CONFIG_FILE
sed -i "s/^CLIENT_DROPOUT_RATE = .*/CLIENT_DROPOUT_RATE = $CLIENT_DROPOUT_RATE/" $CONFIG_FILE
sed -i "s/^ROUND_TIMEOUT_SEC = .*/ROUND_TIMEOUT_SEC = $ROUND_TIMEOUT_SEC/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_RATE = .*/SLOW_SENDER_RATE = $SLOW_SENDER_RATE/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_DELAY_SEC = .*/SLOW_SENDER_DELAY_SEC = $SLOW_SENDER_DELAY_SEC/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_RATE = .*/NETWORK_LATENCY_RATE = $NETWORK_LATENCY_RATE/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_DELAY_SEC = .*/NETWORK_LATENCY_DELAY_SEC = $NETWORK_LATENCY_DELAY_SEC/" $CONFIG_FILE

echo "✅ $CONFIG_FILE updated."


#################################################################
# 3. GENERATE DOCKER-COMPOSE FILE
#################################################################

echo "🔄 Generating docker-compose.yml..."
python generate_compose.py \
  --normal $CLIENTS_NORMAL \
  --slow $CLIENTS_SLOW \
  --lossy $CLIENTS_LOSSY

echo "✅ docker-compose.yml generated."


#################################################################
# 4. EXECUTE SIMULATION
#################################################################

echo "🧹 Cleaning up old containers..."
docker compose down --remove-orphans

mkdir -p fl_logs

LOG_FILE="fl_logs/simulation_$(date +'%Y%m%d_%H%M%S').log"

echo "🚀 Building images..."
docker compose build

echo "📦 Preparing Data (Downloading & Partitioning)..."
docker compose run --rm server python prepare_data.py

echo "▶️ Starting Simulation..."
echo "🪵 Log file will be saved to: $LOG_FILE"

# NEW (Adds '2>&1')
docker compose up --remove-orphans --exit-code-from server 2>&1 | tee $LOG_FILE

echo "---"
echo "✅ Simulation complete. Log saved to $LOG_FILE"
echo "📦 Final model saved to fl_logs/$SAVED_MODEL_NAME"