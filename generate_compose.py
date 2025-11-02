import argparse
import sys

# 1. Define the STATIC header
#    The 'x-client-template' (base class) should NOT have a healthcheck.
#    The 'server:' service MUST have the healthcheck.
YAML_HEADER = """
version: '3.8'

# 1. Define the client template block (NO HEALTHCHECK HERE)
x-client-template: &client-template
  image: fl-framework
  volumes:
    - ./data:/app/data
    - ./fl_logs:/app/fl_logs
  depends_on:
    server:
      condition: service_healthy

services:
  # --- Server Service (HEALTHCHECK GOES HERE) ---
  server:
    build: .
    image: fl-framework
    command: python -m server.app
    ports:
      - "5000:5000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:5000/status"]
      interval: 20s
      timeout: 5s
      retries: 5
      start_period: 60s  # <-- THIS IS THE FIX (was 10s)
    volumes:
      - ./data:/app/data
      - ./fl_logs:/app/fl_logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""

# 2. Define the TEMPLATES for each client profile
# --- HIGH PERFORMANCE CLIENT (GPU + CPU) ---
CLIENT_HIGH_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template  # Inherit the base
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

# --- LOW PERFORMANCE CLIENT (CPU-only, 0.5 cores) ---
CLIENT_LOW_PERF_TEMPLATE = """
  client_{client_id_str}:
    <<: *client-template  # Inherit the base
    command: python -m client.app
    environment:
      - CLIENT_ID=client_{client_id_str}
      - SERVER_URL=http://server:5000
    # No 'deploy' section, so NO GPU
    # Add CPU and Memory limits
    cpus: '0.5'     # Limit to 50% of one CPU core
    mem_limit: '512m' # Limit to 512MB of RAM
"""

def generate_compose_file(num_high, num_low):
    """Generates a docker-compose.yml file for the given clients."""
    
    compose_content = YAML_HEADER
    client_counter = 1

    # Loop to add HIGH_PERF clients
    for _ in range(num_high):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_HIGH_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    # Loop to add LOW_PERF clients
    for _ in range(num_low):
        client_id_str = f"{client_counter:03d}"
        compose_content += CLIENT_LOW_PERF_TEMPLATE.format(client_id_str=client_id_str)
        client_counter += 1

    total_clients = num_high + num_low
    
    try:
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)
        print(f"✅ Successfully generated 'docker-compose.yml' with {total_clients} clients ({num_high} HIGH, {num_low} LOW).")
    except IOError as e:
        print(f"❌ Error writing to file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a docker-compose.yml file for the FL simulation."
    )
    parser.add_argument(
        '--high',
        type=int,
        required=True,
        help="Number of HIGH-performance (GPU) clients."
    )
    parser.add_argument(
        '--low',
        type=int,
        required=True,
        help="Number of LOW-performance (CPU) clients."
    )
    args = parser.parse_args()
    
    if args.high < 0 or args.low < 0 or (args.high + args.low) < 1:
        print("❌ Error: Must have at least one client.")
        sys.exit(1)
        
    generate_compose_file(args.high, args.low)

if __name__ == "__main__":
    main()