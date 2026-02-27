import numpy as np
import random
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
import requests
import time
import json
import sys
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# --- Assumed Imports (from your project) ---
# Make sure these paths are correct for your structure
from client.trainer import train_model 
from client.model_utils import get_model_bytes, set_model_from_bytes
from client.algorithms import get_client_algorithm
from data_utils.loader import get_dataset
from shared.models import get_model
import config

# --- Configuration ---
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000")
CLIENT_ID = os.getenv("CLIENT_ID", "default_client")

# Configure logging
logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Server API Functions ---

def register_client():
    """Registers the client with the server."""
    url = f"{SERVER_URL}/register"
    try:
        response = requests.post(url, json={"client_id": CLIENT_ID})
        if response.status_code == 200:
            logger.info("Successfully registered. Server status: %s", response.json().get("status"))
            return True
        elif response.status_code == 400: # Already registered
             logger.info("Client already registered. Proceeding.")
             return True
        logger.error("Failed to register. Status: %s, Body: %s", response.status_code, response.text)
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to server at %s", url)
        return False

def download_global_model(round_number):
    """Downloads the global model from the server for a specific round."""
    url = f"{SERVER_URL}/download_model"
    try:
        if config.NETWORK_LATENCY_RATE > 0 and random.random() < config.NETWORK_LATENCY_RATE:
            delay = config.NETWORK_LATENCY_DELAY_SEC
            logger.warning(f"SIMULATING NETWORK LATENCY: Delaying download by {delay}s...")
            time.sleep(delay)
        response = requests.get(url, params={"round": round_number})
        if response.status_code == 200:
            logger.info(f"Downloaded model for round {round_number} ({len(response.content)} bytes).")
            return response.content
        logger.error(f"Failed download. Status: {response.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to server at %s", url)
        return None

def submit_model_update(model, num_samples, metrics):
    """Submits the locally trained model update to the server."""
    url = f"{SERVER_URL}/submit_update"
    
    try:
        # 1. Prepare Binary Data
        # Start with standard model weights
        binary_data = model.state_dict()
        
        # Check if we have Tensor metrics (like SCAFFOLD's delta_c)
        tensor_metrics = {}
        keys_to_remove = []
        
        for key, value in metrics.items():
            # Check if it's a Tensor or a dict of Tensors
            if hasattr(value, 'cpu') or (isinstance(value, dict) and len(value) > 0 and hasattr(next(iter(value.values())), 'cpu')):
                tensor_metrics[key] = value
                keys_to_remove.append(key)
        
        # Remove Tensors from the JSON metrics dict to prevent serialization errors
        for k in keys_to_remove:
            del metrics[k]

        # If we found tensors, create a Composite Payload
        if tensor_metrics:
            binary_data = {
                "model_state": binary_data,
                "tensor_metrics": tensor_metrics
            }
            
        # Serialize the binary data
        model_bytes = get_model_bytes(binary_data)
        
        # 2. Update Upload Size Metric
        metrics["payload_size_mb"] = len(model_bytes) / (1024 * 1024)
        
        # 3. Prepare Metadata (Safe JSON)
        metadata = {
            "client_id": CLIENT_ID,
            "num_samples": num_samples,
            "metrics": metrics 
        }
        
        # 4. Send Multipart Request
        files = {
            'model': ('model.pth', model_bytes, 'application/octet-stream'),
            'json': (None, json.dumps(metadata), 'application/json'),
        }

        # Simulation: Slow Sender
        if config.SLOW_SENDER_RATE > 0 and random.random() < config.SLOW_SENDER_RATE:
            delay = config.SLOW_SENDER_DELAY_SEC
            logger.warning(f"SIMULATING SLOW SENDER: Delaying by {delay}s...")
            time.sleep(delay)

        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            logger.info("Successfully submitted model update.")
            return True
        
        logger.error(f"Failed to submit update. Status: {response.status_code}, Body: {response.text}")
        return False

    except Exception as e:
        logger.error(f"Error during model submission: {e}", exc_info=True)
        return False

def check_server_status():
    """Polls the server for its current status and round."""
    url = f"{SERVER_URL}/status"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        logger.warning("Could not get server status: %s", response.status_code)
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Server connection failed during status check.")
        return None

def get_client_dataloader(client_id_num):
    """
    Loads the configured dataset and returns a DataLoader using pre-calculated
    indices from partitions.json.
    """
    logger.info(f"Loading {config.DATASET_NAME} data for client {client_id_num}...")

    # 1. Load the dataset using the factory (DO NOT DOWNLOAD - It should exist)
    # The factory handles the transforms automatically now.
    try:
        train_dataset = get_dataset(config.DATASET_NAME, train=True, download=False)
    except Exception as e:
        logger.error(f"{config.DATASET_NAME} dataset not found in ./data! Did you run prepare_data.py? Error: {e}")
        raise

    # 2. Load the pre-calculated partition indices
    partition_file = './data/partitions.json'
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file {partition_file} not found.")
    
    with open(partition_file, 'r') as f:
        partitions = json.load(f)
    
    # Get indices for THIS client (keys are strings in JSON)
    client_indices = partitions.get(str(client_id_num))
    if client_indices is None:
        raise ValueError(f"No partition found for client {client_id_num}")

    # 3. Create the Subset and DataLoader
    client_subset = Subset(train_dataset, client_indices)
    
    client_loader = DataLoader(client_subset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=2)
    
    logger.info(f"Client {CLIENT_ID} loaded {len(client_subset)} samples from partitions.json.")
    return client_loader

# --- Main Client Loop ---

def main():
    logger.info("--- Starting FL Client %s ---", CLIENT_ID)
    
    # 1. Register with the server
    if not register_client():
        logger.error("Could not register with server. Exiting.")
        return

    # --- Instantiate your REAL model ---
    global_model = get_model(config.MODEL_NAME)
    logger.info("Model %s instantiated.", global_model.__class__.__name__)
    
    # --- Instantiate the Algorithm ONCE (Preserves State) ---
    algorithm = get_client_algorithm(config.CLIENT_ALGO)
    logger.info(f"Algorithm {config.CLIENT_ALGO} initialized.")
    
    # --- Load this client's data partition ---
    try:
        # Get the number from the CLIENT_ID (e.g., "client_001" -> 1)
        client_id_num = int(CLIENT_ID.split('_')[-1])
        if client_id_num > config.TOTAL_CLIENTS or client_id_num < 1:
            raise ValueError(f"CLIENT_ID {CLIENT_ID} not in expected range 1-{config.TOTAL_CLIENTS}")
        
        client_dataloader = get_client_dataloader(client_id_num)
    except Exception as e:
        logger.error(f"Failed to load data: {e}. Exiting.")
        return
    
    current_round = 0

    # 2. Download initial model (Round 0)
    logger.info("Waiting for initial model (round 0)...")
    model_bytes = None
    while model_bytes is None:
        model_bytes = download_global_model(current_round)
        if model_bytes is None:
            time.sleep(5)

    # ===================================================================
    # Main Training Loop
    # ===================================================================
    while True:
        logger.info("--- Starting Round %d ---", current_round)
        
        # 3. Download and Extract
        try:
            # set_model_from_bytes now returns the extra payload (e.g., Global C for Scaffold)
            extra_payload = set_model_from_bytes(global_model, model_bytes)
            
            # 4. Train the model using the algorithm
            logger.info("Starting local training...")
            start_time = time.time()
            
            # Pass algorithm instance and extra payload to train_model
            num_samples, peak_gpu_mb, peak_ram_mb, training_metrics = train_model(
                global_model, 
                client_dataloader, 
                algorithm, 
                extra_payload
            )
            
            end_time = time.time()
            training_time_sec = end_time - start_time
            logger.info(f"Local training complete in {training_time_sec:.2f}s.")

        except Exception as e:
            logger.error(f"Local training for round {current_round} failed: {e}", exc_info=True)
            time.sleep(15) 
            continue
        
        # 5. Submit model update
        logger.info("Submitting trained model update from %d samples...", num_samples)
        
        # Combine all metrics
        client_metrics = {
            "training_time_sec": training_time_sec,
            "peak_gpu_mb": peak_gpu_mb,
            "peak_ram_mb": peak_ram_mb,
            **training_metrics  # Include any additional metrics from training
        }
        
        if not submit_model_update(global_model, num_samples, client_metrics):
            logger.warning("Failed to submit update. Retrying in 10s.")
            time.sleep(10)
            continue 

        logger.info("Update submitted. Waiting for next round...")
        
        # 6. Wait for server to finish aggregation
        while True:
            time.sleep(config.POLL_INTERVAL)
            status_data = check_server_status()
            
            if not status_data:
                logger.warning("Could not reach server. Retrying...")
                continue
                
            server_status = status_data.get("status")
            server_round = status_data.get("current_round")

            if server_status == "TRAINING_COMPLETE":
                logger.info("--- Server reports training complete. Exiting. ---")
                return 

            if server_round > current_round and server_status == "WAITING":
                logger.info("Server has new model for round %d. Downloading...", server_round)
                
                new_model_bytes_response = download_global_model(server_round)
                
                if new_model_bytes_response:
                    model_bytes = new_model_bytes_response
                    current_round = server_round  
                    break 
                else:
                    logger.warning("Failed to download new model. Retrying...")

if __name__ == "__main__":
    main()