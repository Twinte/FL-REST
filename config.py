# --- Server Configuration ---
MODEL_NAME = "SimpleCNN"
AGGREGATION_STRATEGY = "FedAvgM"
SERVER_LEARNING_RATE = 1.0
SERVER_MOMENTUM = 0.9
DEVICE = "auto"
TOTAL_ROUNDS = 10
MIN_CLIENTS_PER_ROUND = 5
MIN_CLIENTS_FOR_AGGREGATION = 3
SAVED_MODEL_NAME = "final_global_model.pth"

# --- Client Algorithm (How we train locally) ---
# Options: "Standard", "FedProx"
CLIENT_ALGO = "FedProx"
FEDPROX_MU = 0.01

# --- Client Configuration ---
TOTAL_CLIENTS = 5
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9
POLL_INTERVAL = 10

# --- Data Configuration (Non-IID) ---
DIRICHLET_ALPHA = 0.5
RANDOM_SEED = 42

# --- Simulation of FL Conditions ---
CLIENT_DROPOUT_RATE = 0.0
ROUND_TIMEOUT_SEC = 300
# --- NEW: Slow Sender & Latency Simulation ---
SLOW_SENDER_RATE = 0.0
SLOW_SENDER_DELAY_SEC = 30
NETWORK_LATENCY_RATE = 0.0
NETWORK_LATENCY_DELAY_SEC = 5

# --- Robustness ---
FEDPROX_MU = 0.01