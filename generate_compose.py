import argparse
import sys

# List of environment variables to pass from host to containers
# These match the variables exported in run.sh
ENV_VARS = [
    "MODEL_NAME", "AGGREGATION_STRATEGY", "SERVER_LEARNING_RATE", "SERVER_MOMENTUM",
    "DEVICE", "TOTAL_ROUNDS", "MIN_CLIENTS_PER_ROUND", "MIN_CLIENTS_FOR_AGGREGATION",
    "SAVED_MODEL_NAME", "CLIENT_ALGO", "FEDPROX_MU", "TOTAL_CLIENTS",
    "LOCAL_EPOCHS", "BATCH_SIZE", "LEARNING_RATE", "MOMENTUM", "POLL_INTERVAL",
    "DIRICHLET_ALPHA", "RANDOM_SEED", "CLIENT_DROPOUT_RATE", "ROUND_TIMEOUT_SEC",
    "SLOW_SENDER_RATE", "SLOW_SENDER_DELAY_SEC", "NETWORK_LATENCY_RATE",
    "NETWORK_LATENCY_DELAY_SEC"
]

def get_env_lines(indent_level):
    """Generates the list of environment variables formatted for YAML."""
    indent = " " * indent_level
    lines = ""
    for var in ENV_VARS:
        # Syntax: - VAR=${VAR}
        lines += f"{indent}- {var}=${{{var}}}\n"
    return lines

# Pre-generate server environment lines (indentation 6 to match 'environment:' at 4)
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

# Network profiles for 'tc' command
TC_COMMANDS = {
    "normal": "",
    "slow": "tc qdisc add dev eth0 root tbf rate 5mbit burst 32kbit latency 400ms",
    "fast_latency": "tc qdisc add dev eth0 root netem delay 100ms 20ms",
    "lossy": "tc qdisc add dev eth0 root netem loss 5%",
    "mobile": "tc qdisc add dev eth0 root netem delay 50ms 10ms rate 20mbit"
}

def generate_client_service(client_id, profile="normal"):
    client_id_str = f"{client_id:03d}"
    tc_cmd = TC_COMMANDS.get(profile, "")
    
    if tc_cmd:
        command = (
            f'sh -c "tc qdisc del dev eth0 root 2>/dev/null || true && '
            f'{tc_cmd} && '
            f'echo \'[Network] Applied profile: {profile}\' && '
            f'python -m client.app"'
        )
    else:
        command = "python -m client.app"

    # Hardware resources
    if profile in ["slow", "mobile"]:
        resources = "    cpus: '0.5'\n    mem_limit: '512m'"
        gpu_deploy = ""
    else:
        resources = ""
        gpu_deploy = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]"""

    # We inject the GLOBAL vars here so they don't get overwritten
    global_env_lines = get_env_lines(6)

    return f"""
  client_{client_id_str}:
    <<: *client-template
    command: >
      {command}
    environment:
{global_env_lines}      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
      - NETWORK_PROFILE={profile}
{resources}{gpu_deploy}
"""

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose with network profiles.")
    parser.add_argument('--normal', type=int, default=0)
    parser.add_argument('--slow', type=int, default=0)
    parser.add_argument('--lossy', type=int, default=0)
    args = parser.parse_args()
    
    total_clients = args.normal + args.slow + args.lossy
    if total_clients < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)

    compose_content = YAML_HEADER
    client_counter = 1

    for _ in range(args.normal):
        compose_content += generate_client_service(client_counter, "normal")
        client_counter += 1
        
    for _ in range(args.slow):
        compose_content += generate_client_service(client_counter, "slow")
        client_counter += 1

    for _ in range(args.lossy):
        compose_content += generate_client_service(client_counter, "lossy")
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