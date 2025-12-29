import argparse
import sys

# 1. Static Header (Server + Template)
YAML_HEADER = """
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
  # Enable NET_ADMIN for all clients so they can use 'tc'
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
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

# 2. Network Profiles Definition
TC_COMMANDS = {
    "normal": "",  # No restriction
    "slow": "tc qdisc add dev eth0 root tbf rate 5mbit burst 32kbit latency 400ms",
    "fast_latency": "tc qdisc add dev eth0 root netem delay 100ms 20ms", # 100ms delay ±20ms jitter
    "lossy": "tc qdisc add dev eth0 root netem loss 5%", # 5% packet loss
    "mobile": "tc qdisc add dev eth0 root netem delay 50ms 10ms rate 20mbit"
}

def generate_client_service(client_id, profile="normal"):
    client_id_str = f"{client_id:03d}"
    tc_cmd = TC_COMMANDS.get(profile, "")
    
    # Construct the startup command
    if tc_cmd:
        # We use a shell to chain: clean old rules -> apply new rules -> start app
        # "|| true" ensures we don't crash if no rules existed previously
        command = (
            f'sh -c "tc qdisc del dev eth0 root 2>/dev/null || true && '
            f'{tc_cmd} && '
            f'echo \'[Network] Applied profile: {profile}\' && '
            f'python -m client.app"'
        )
    else:
        command = "python -m client.app"

    # Hardware resource allocation based on profile
    # (You can customize this: e.g., 'slow' clients get less CPU)
    if profile in ["slow", "mobile"]:
        resources = "    cpus: '0.5'\n    mem_limit: '512m'"
        gpu_deploy = "" # No GPU for slow devices
    else:
        resources = ""
        # Give GPU to normal/lossy/fast clients
        gpu_deploy = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]"""

    return f"""
  client_{client_id_str}:
    <<: *client-template
    command: >
      {command}
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
      - NETWORK_PROFILE={profile}
{resources}{gpu_deploy}
"""

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose with network profiles.")
    parser.add_argument('--normal', type=int, default=0, help="Count of normal clients")
    parser.add_argument('--slow', type=int, default=0, help="Count of slow/throttled clients")
    parser.add_argument('--lossy', type=int, default=0, help="Count of clients with packet loss")
    args = parser.parse_args()
    
    total_clients = args.normal + args.slow + args.lossy
    if total_clients < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)

    compose_content = YAML_HEADER
    client_counter = 1

    # Generate entries for each type
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
        print(f"   - Normal: {args.normal}")
        print(f"   - Slow:   {args.slow}")
        print(f"   - Lossy:  {args.lossy}")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()