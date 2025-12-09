import argparse
import sys

# 1. Define the STATIC header (REMOVED docker-tc service)
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

services:
  # docker-tc removed

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

# 2. Define the TEMPLATES (REMOVED labels section)

# --- HIGH PERFORMANCE CLIENT (Fast Network) ---
CLIENT_HIGH_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template
    command: python -m client.app
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

# --- LOW PERFORMANCE CLIENT (Slow Network) ---
CLIENT_LOW_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template
    command: python -m client.app
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
    cpus: '0.5'
    mem_limit: '512m'
"""

def generate_compose_file(num_high, num_low):
    compose_content = YAML_HEADER
    client_counter = 1

    for _ in range(num_high):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_HIGH_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    for _ in range(num_low):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_LOW_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    total_clients = num_high + num_low
    
    try:
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        print(f"✅ Successfully generated 'docker-compose.yml' with {total_clients} clients (No Docker-TC).")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose.")
    parser.add_argument('--high', type=int, required=True)
    parser.add_argument('--low', type=int, required=True)
    args = parser.parse_args()
    
    if args.high < 0 or args.low < 0 or (args.high + args.low) < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)
        
    generate_compose_file(args.high, args.low)

if __name__ == "__main__":
    main()