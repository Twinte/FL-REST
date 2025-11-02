#!/bin/bash
set -e # Exit immediately if any command fails

#################################################################
# 1. SIMULATION PARAMETERS
# (Edit these values to configure your test run)
#################################################################

# --- Client & Round Config ---
# Define how many of each client type to create
CLIENTS_HIGH_PERF=4  # Gets GPU + high CPU
CLIENTS_LOW_PERF=0   # Gets NO GPU + limited CPU

TOTAL_ROUNDS=5
MIN_CLIENTS_FOR_AGGREGATION=4
# ---------------------------------------------------------------
# (Derived values - DO NOT EDIT)
NUM_CLIENTS=$(($CLIENTS_HIGH_PERF + $CLIENTS_LOW_PERF))
MIN_CLIENTS_PER_ROUND=$NUM_CLIENTS

# --- Client Training Config ---
LOCAL_EPOCHS=3
BATCH_SIZE=32
LEARNING_RATE=0.01

# --- Data Config (Non-IID) ---
DIRICHLET_ALPHA=0.5

# --- FL Condition Simulation ---
CLIENT_DROPOUT_RATE=0.0  # 0.0 = no dropout, 0.3 = 30% dropout
ROUND_TIMEOUT_SEC=300     # Max time in seconds for a round

# --- NEW: Slow Sender & Latency Config ---
SLOW_SENDER_RATE=0.0
SLOW_SENDER_DELAY_SEC=30
NETWORK_LATENCY_RATE=0.0
NETWORK_LATENCY_DELAY_SEC=5

# --- Other System Config ---
DEVICE="auto" # Options: "auto", "cpu", "cuda"
MOMENTUM=0.9
POLL_INTERVAL=10
RANDOM_SEED=42
SAVED_MODEL_NAME="final_global_model.pth"

# ---------------------------------------------------------------
# (Derived values - DO NOT EDIT)
MIN_CLIENTS_PER_ROUND=$NUM_CLIENTS
TOTAL_CLIENTS=$NUM_CLIENTS
CONFIG_FILE="config.py"
# ---------------------------------------------------------------


#################################################################
# 2. CONFIGURATION SCRIPT
# (This section automatically updates config.py)
#################################################################

echo "▶️ Starting simulation with $NUM_CLIENTS clients for $TOTAL_ROUNDS rounds..."

# Check if config.py exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found. Cannot configure run."
    exit 1
fi

# Use 'sed' to find and replace values in config.py
# This command finds the line starting with "VAR_NAME =" and replaces it.
echo "🔄 Updating $CONFIG_FILE with new parameters..."
sed -i "s/^TOTAL_ROUNDS = .*/TOTAL_ROUNDS = $TOTAL_ROUNDS/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_PER_ROUND = .*/MIN_CLIENTS_PER_ROUND = $MIN_CLIENTS_PER_ROUND/" $CONFIG_FILE
sed -i "s/^TOTAL_CLIENTS = .*/TOTAL_CLIENTS = $TOTAL_CLIENTS/" $CONFIG_FILE
sed -i "s/^LOCAL_EPOCHS = .*/LOCAL_EPOCHS = $LOCAL_EPOCHS/" $CONFIG_FILE
sed -i "s/^BATCH_SIZE = .*/BATCH_SIZE = $BATCH_SIZE/" $CONFIG_FILE
sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = $LEARNING_RATE/" $CONFIG_FILE
sed -i "s/^MOMENTUM = .*/MOMENTUM = $MOMENTUM/" $CONFIG_FILE
sed -i "s/^POLL_INTERVAL = .*/POLL_INTERVAL = $POLL_INTERVAL/" $CONFIG_FILE
sed -i "s/^DIRICHLET_ALPHA = .*/DIRICHLET_ALPHA = $DIRICHLET_ALPHA/" $CONFIG_FILE
sed -i "s/^RANDOM_SEED = .*/RANDOM_SEED = $RANDOM_SEED/" $CONFIG_FILE
sed -i "s/^SAVED_MODEL_NAME = .*/SAVED_MODEL_NAME = \"$SAVED_MODEL_NAME\"/" $CONFIG_FILE
sed -i "s/^DEVICE = .*/DEVICE = \"$DEVICE\"/" $CONFIG_FILE
sed -i "s/^CLIENT_DROPOUT_RATE = .*/CLIENT_DROPOUT_RATE = $CLIENT_DROPOUT_RATE/" $CONFIG_FILE
sed -i "s/^ROUND_TIMEOUT_SEC = .*/ROUND_TIMEOUT_SEC = $ROUND_TIMEOUT_SEC/" $CONFIG_FILE
sed -i "s/^MIN_CLIENTS_FOR_AGGREGATION = .*/MIN_CLIENTS_FOR_AGGREGATION = $MIN_CLIENTS_FOR_AGGREGATION/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_RATE = .*/SLOW_SENDER_RATE = $SLOW_SENDER_RATE/" $CONFIG_FILE
sed -i "s/^SLOW_SENDER_DELAY_SEC = .*/SLOW_SENDER_DELAY_SEC = $SLOW_SENDER_DELAY_SEC/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_RATE = .*/NETWORK_LATENCY_RATE = $NETWORK_LATENCY_RATE/" $CONFIG_FILE
sed -i "s/^NETWORK_LATENCY_DELAY_SEC = .*/NETWORK_LATENCY_DELAY_SEC = $NETWORK_LATENCY_DELAY_SEC/" $CONFIG_FILE

echo "✅ $CONFIG_FILE updated."


#################################################################
# 3. GENERATE DOCKER-COMPOSE FILE
#################################################################

echo "🔄 Generating docker-compose.yml for $NUM_CLIENTS clients..."
python generate_compose.py --high $CLIENTS_HIGH_PERF --low $CLIENTS_LOW_PERF
echo "✅ docker-compose.yml generated."


#################################################################
# 4. EXECUTE SIMULATION
#################################################################

echo "🧹 Cleaning up old containers..."
docker-compose down

# Create a unique log file name with a timestamp
LOG_FILE="fl_logs/simulation_$(date +'%Y%m%d_%H%M%S').log"

echo "🚀 Building and running simulation..."
echo "🪵 Log file will be saved to: $LOG_FILE"

# Run the simulation
# The '| tee' command shows logs in the terminal AND saves to a file.
docker-compose up --build --exit-code-from server | tee $LOG_FILE

echo "---"
echo "✅ Simulation complete. Log saved to $LOG_FILE"
echo "📦 Final model saved to fl_logs/$SAVED_MODEL_NAME"