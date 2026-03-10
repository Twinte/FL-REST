import os

def get_env(key, default, cast_type=str):
    """Helper to read env vars with type casting."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return cast_type(value)
    except ValueError:
        return default

# --- Server Configuration ---
MODEL_NAME = get_env("MODEL_NAME", "SimpleCNN")
AGGREGATION_STRATEGY = get_env("AGGREGATION_STRATEGY", "FedAvgM")
SERVER_LEARNING_RATE = get_env("SERVER_LEARNING_RATE", 1.0, float)
SERVER_MOMENTUM = get_env("SERVER_MOMENTUM", 0.9, float)
DEVICE = get_env("DEVICE", "auto")
TOTAL_ROUNDS = get_env("TOTAL_ROUNDS", 10, int)
MIN_CLIENTS_PER_ROUND = get_env("MIN_CLIENTS_PER_ROUND", 5, int)
MIN_CLIENTS_FOR_AGGREGATION = get_env("MIN_CLIENTS_FOR_AGGREGATION", 3, int)
SAVED_MODEL_NAME = get_env("SAVED_MODEL_NAME", "final_global_model.pth")

# --- Client Algorithm ---
CLIENT_ALGO = get_env("CLIENT_ALGO", "FedProx")
FEDPROX_MU = get_env("FEDPROX_MU", 0.01, float)

# --- Client Configuration ---
TOTAL_CLIENTS = get_env("TOTAL_CLIENTS", 5, int)
LOCAL_EPOCHS = get_env("LOCAL_EPOCHS", 3, int)
BATCH_SIZE = get_env("BATCH_SIZE", 32, int)
LEARNING_RATE = get_env("LEARNING_RATE", 0.01, float)
MOMENTUM = get_env("MOMENTUM", 0.9, float)
POLL_INTERVAL = get_env("POLL_INTERVAL", 10, int)

# --- Data Configuration ---
DATASET_NAME = get_env("DATASET_NAME", "CIFAR10")  # CIFAR10 or CIFAR100
DIRICHLET_ALPHA = get_env("DIRICHLET_ALPHA", 0.5, float)
RANDOM_SEED = get_env("RANDOM_SEED", 42, int)

# --- Simulation of FL Conditions ---
CLIENT_DROPOUT_RATE = get_env("CLIENT_DROPOUT_RATE", 0.0, float)
ROUND_TIMEOUT_SEC = get_env("ROUND_TIMEOUT_SEC", 300, int)

# --- Network Traffic Simulation ---
SLOW_SENDER_RATE = get_env("SLOW_SENDER_RATE", 0.0, float)
SLOW_SENDER_DELAY_SEC = get_env("SLOW_SENDER_DELAY_SEC", 30, int)
NETWORK_LATENCY_RATE = get_env("NETWORK_LATENCY_RATE", 0.0, float)
NETWORK_LATENCY_DELAY_SEC = get_env("NETWORK_LATENCY_DELAY_SEC", 5, int)

# =============================================================================
# FedPrune Configuration
# =============================================================================

# EMA decay for importance tracking (higher = smoother, slower to adapt)
EMA_DECAY = get_env("EMA_DECAY", 0.9, float)

# Weight for magnitude vs variance in combined importance score
# 0.3 = variance-heavy (Neyman Allocation priority), 0.7 = magnitude-heavy
IMPORTANCE_ALPHA = get_env("IMPORTANCE_ALPHA", 0.3, float)

# FLuID: clients with capacity >= this threshold become leaders (train full model)
FLUID_LEADER_THRESHOLD = get_env("FLUID_LEADER_THRESHOLD", 0.7, float)

# Per-client capacity ratio (set per container by generate_compose.py).
# Each client reads its own capacity from this env var and reports it
# to the server during registration. Default 0.5 if not set.
CLIENT_CAPACITY = get_env("CLIENT_CAPACITY", 0.5, float)