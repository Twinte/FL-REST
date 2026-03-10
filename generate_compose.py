"""
Docker Compose Generator for FL-REST
======================================
Generates docker-compose.yml with client containers configured for
different experiment types.

Two profile families:

  NETWORK PROFILES (for bandwidth/robustness experiments):
    --normal, --slow, --lossy
    Controls: tc qdisc rules (bandwidth caps, packet loss)
    Used by: run.sh with FedAvg/FedProx/Scaffold

  DEVICE PROFILES (for compute-heterogeneity / FedPrune experiments):
    --high_perf, --mid_perf, --low_perf
    Controls: CPU cores, memory limits, GPU access, FedPrune capacity
    Used by: run_fedprune.sh with FedPrune strategy

You can use one family or mix both (e.g., a low_perf device on a lossy link),
but typically you'd use one set per experiment.
"""

import argparse
import sys
import os

# =============================================================================
# Environment variables passed from host → containers
# =============================================================================

ENV_VARS = [
    "MODEL_NAME", "AGGREGATION_STRATEGY", "SERVER_LEARNING_RATE", "SERVER_MOMENTUM",
    "DEVICE", "TOTAL_ROUNDS", "MIN_CLIENTS_PER_ROUND", "MIN_CLIENTS_FOR_AGGREGATION",
    "SAVED_MODEL_NAME", "CLIENT_ALGO", "FEDPROX_MU", "TOTAL_CLIENTS",
    "LOCAL_EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "MOMENTUM", "POLL_INTERVAL",
    "DIRICHLET_ALPHA", "RANDOM_SEED", "CLIENT_DROPOUT_RATE", "ROUND_TIMEOUT_SEC",
    "SLOW_SENDER_RATE", "SLOW_SENDER_DELAY_SEC", "NETWORK_LATENCY_RATE",
    "NETWORK_LATENCY_DELAY_SEC",
    # FedPrune
    "EMA_DECAY", "IMPORTANCE_ALPHA",
    # Dataset
    "DATASET_NAME",
]

def get_env_lines(indent_level):
    """Generates the list of environment variables formatted for YAML."""
    indent = " " * indent_level
    lines = ""
    for var in ENV_VARS:
        lines += f"{indent}- {var}=${{{var}}}\n"
    return lines

server_env_lines = get_env_lines(6)

YAML_HEADER = f"""
version: '3.8'

x-client-template: &client-template
  image: fl-framework
  volumes:
    - ./data:/app/data
    - ./fl_logs:/app/fl_logs
    - ./config.py:/app/config.py
  depends_on:
    server:
      condition: service_healthy
  cap_add:
    - NET_ADMIN

services:
  server:
    build: .
    image: fl-framework
    command: >
      sh -c "tensorboard --logdir=/app/fl_logs/tensorboard --port=6006 --host=0.0.0.0 &
             python -m server.app"
    ports:
      - "5000:5000"
      - "6006:6006"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:5000/status"]
      interval: 20s
      timeout: 5s
      retries: 5
      start_period: 60s
    volumes:
      - ./data:/app/data
      - ./fl_logs:/app/fl_logs
      - ./config.py:/app/config.py
    environment:
{server_env_lines}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""


# =============================================================================
# Network Profiles (existing — for bandwidth/robustness experiments)
# =============================================================================

TC_COMMANDS = {
    "normal": "",
    "slow": "tc qdisc add dev eth0 root tbf rate 5mbit burst 32kbit latency 400ms",
    "fast_latency": "tc qdisc add dev eth0 root netem delay 100ms 20ms",
    "lossy": "tc qdisc add dev eth0 root netem loss 5%",
    "mobile": "tc qdisc add dev eth0 root netem delay 50ms 10ms rate 20mbit",
}


# =============================================================================
# Device Profiles (new — for compute-heterogeneity / FedPrune experiments)
# =============================================================================
# Each profile models a class of real-world device:
#
#   high_perf  → Edge server / workstation with GPU
#                Full CPU, 2GB RAM, GPU access
#                FedPrune capacity: 0.7 (trains 70% of neurons)
#
#   mid_perf   → Standard laptop / desktop
#                Moderate CPU (1.5 cores), 1GB RAM, GPU access
#                FedPrune capacity: 0.5 (trains 50% of neurons)
#
#   low_perf   → Mobile / IoT / embedded device
#                Limited CPU (0.5 cores), 512MB RAM, CPU-only
#                FedPrune capacity: 0.3 (trains 30% of neurons)

DEVICE_PROFILES = {
    "high_perf": {
        "capacity": 0.7,
        "cpus": None,         # No CPU limit (full access)
        "mem_limit": "2g",
        "gpu": True,
        "description": "Edge server (GPU, full CPU, 2GB)",
    },
    "mid_perf": {
        "capacity": 0.5,
        "cpus": "1.5",
        "mem_limit": "1g",
        "gpu": True,
        "description": "Standard device (GPU, 1.5 CPU, 1GB)",
    },
    "low_perf": {
        "capacity": 0.3,
        "cpus": "0.5",
        "mem_limit": "512m",
        "gpu": False,
        "description": "IoT/mobile (CPU-only, 0.5 CPU, 512MB)",
    },
}

# Allow env var overrides for capacity per profile
def get_profile_capacity(profile):
    """Get capacity from env var override or device profile default."""
    env_key = f"CAPACITY_{profile.upper()}"
    env_val = os.getenv(env_key)
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            pass
    if profile in DEVICE_PROFILES:
        return DEVICE_PROFILES[profile]["capacity"]
    # Fallback for old network-only profiles
    return 0.5


# =============================================================================
# Service Generator
# =============================================================================

def generate_client_service(client_id, profile="normal"):
    """
    Generate a docker-compose service block for a single client.
    
    Handles both legacy network profiles and new device profiles.
    """
    client_id_str = f"{client_id:03d}"
    
    # --- Determine if this is a device profile or network profile ---
    is_device_profile = profile in DEVICE_PROFILES
    device_cfg = DEVICE_PROFILES.get(profile, None)
    
    # --- Network shaping (tc commands) ---
    tc_cmd = TC_COMMANDS.get(profile, "")
    
    if tc_cmd:
        command = (
            f'sh -c "tc qdisc del dev eth0 root 2>/dev/null || true && '
            f'{tc_cmd} && '
            f'echo \'[Network] Applied profile: {profile}\' && '
            f'python -m client.app"'
        )
    elif is_device_profile:
        command = (
            f'sh -c "echo \'[Device] Applied profile: {profile} '
            f'(capacity={get_profile_capacity(profile)})\' && '
            f'python -m client.app"'
        )
    else:
        command = "python -m client.app"

    # --- Resource constraints ---
    resource_lines = ""
    gpu_deploy = ""
    
    if is_device_profile:
        # Device profile: constraints from profile definition
        constraints = []
        if device_cfg["cpus"] is not None:
            constraints.append(f"    cpus: '{device_cfg['cpus']}'")
        if device_cfg["mem_limit"] is not None:
            constraints.append(f"    mem_limit: '{device_cfg['mem_limit']}'")
        resource_lines = "\n".join(constraints) if constraints else ""
        
        if device_cfg["gpu"]:
            gpu_deploy = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]"""
    else:
        # Legacy network profile: original behavior
        if profile in ["slow", "mobile"]:
            resource_lines = "    cpus: '0.5'\n    mem_limit: '512m'"
        else:
            gpu_deploy = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]"""

    # --- Capacity for FedPrune ---
    capacity = get_profile_capacity(profile)

    # --- Environment variables ---
    global_env_lines = get_env_lines(6)

    return f"""
  client_{client_id_str}:
    <<: *client-template
    command: >
      {command}
    environment:
{global_env_lines}      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
      - DEVICE_PROFILE={profile}
      - CLIENT_CAPACITY={capacity}
{resource_lines}{gpu_deploy}
"""


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate docker-compose.yml with client profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FedPrune experiment with mixed device capacities:
  python generate_compose.py --high_perf 1 --mid_perf 2 --low_perf 2

  # Network robustness experiment (original profiles):
  python generate_compose.py --normal 2 --slow 2 --lossy 1

  # Mixed: device heterogeneity + network conditions:
  python generate_compose.py --high_perf 1 --mid_perf 2 --slow 1 --lossy 1
        """
    )
    
    # Device profiles (FedPrune)
    parser.add_argument('--high_perf', type=int, default=0,
                        help='High-perf devices (GPU, full CPU, capacity=0.7)')
    parser.add_argument('--mid_perf', type=int, default=0,
                        help='Mid-range devices (GPU, moderate CPU, capacity=0.5)')
    parser.add_argument('--low_perf', type=int, default=0,
                        help='Low-end devices (CPU-only, limited, capacity=0.3)')
    
    # Network profiles (legacy, backward-compatible)
    parser.add_argument('--normal', type=int, default=0,
                        help='Normal network clients (no constraints)')
    parser.add_argument('--slow', type=int, default=0,
                        help='Slow network clients (5mbit cap)')
    parser.add_argument('--lossy', type=int, default=0,
                        help='Lossy network clients (5%% packet loss)')
    
    args = parser.parse_args()
    
    # Build ordered list of (count, profile_name) pairs
    profile_order = [
        (args.high_perf, "high_perf"),
        (args.mid_perf, "mid_perf"),
        (args.low_perf, "low_perf"),
        (args.normal, "normal"),
        (args.slow, "slow"),
        (args.lossy, "lossy"),
    ]
    
    total_clients = sum(count for count, _ in profile_order)
    if total_clients < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)

    compose_content = YAML_HEADER
    client_counter = 1

    print(f"📊 Client profiles ({total_clients} total):")
    
    for count, profile in profile_order:
        for _ in range(count):
            compose_content += generate_client_service(client_counter, profile)
            cap = get_profile_capacity(profile)
            if profile in DEVICE_PROFILES:
                desc = DEVICE_PROFILES[profile]["description"]
                print(f"   client_{client_counter:03d} → {profile:10s} capacity={cap:.1f}  ({desc})")
            else:
                print(f"   client_{client_counter:03d} → {profile:10s} capacity={cap:.1f}  (network profile)")
            client_counter += 1

    try:
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        print(f"✅ Generated docker-compose.yml with {total_clients} clients.")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()