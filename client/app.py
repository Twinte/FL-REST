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
from client.model_utils import get_model_state, set_model_state
from client.model import get_model
import config

# --- Configuration ---
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:5000")
CLIENT_ID = os.getenv("CLIENT_ID", "default_client")
#POLL_INTERVAL = 10 # Seconds to wait between status checks

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
        # --- CHANGE THIS LINE ---
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
            logger.info("Successfully downloaded global model for round %d.", round_number)
            model_state = response.json().get("model_state")
            return model_state
        # --- CHANGE THIS LINE ---
        logger.error("Failed to download model. Status: %s, Body: %s", response.status_code, response.text)
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to server at %s", url)
        return None

def submit_model_update(model_state, num_samples, metrics):
    """Submits the locally trained model update to the server."""
    url = f"{SERVER_URL}/submit_update"
    payload = {
        "client_id": CLIENT_ID,
        "model_update": model_state,
        "num_samples": num_samples,
        "metrics": metrics
    }
    try:
        payload_str = json.dumps(payload)
        metrics["payload_size_mb"] = len(payload_str.encode('utf-8')) / (1024 * 1024)
        # Update the payload string *with* the new size
        payload["metrics"] = metrics
        payload_str = json.dumps(payload)
    except Exception as e:
        logger.warning(f"Could not calculate payload size: {e}")
        # We can still proceed without this metric
        payload_str = json.dumps(payload)

    try:
        if config.SLOW_SENDER_RATE > 0 and random.random() < config.SLOW_SENDER_RATE:
            delay = config.SLOW_SENDER_DELAY_SEC
            logger.warning(f"SIMULATING SLOW SENDER: Delaying update submission by {delay}s...")
            time.sleep(delay)
        response = requests.post(url, data=payload_str, headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            logger.info("Successfully submitted model update.")
            return True
        logger.error("Failed to submit update. Status: %s, Body: %s", response.status_code, response.text)
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to server at %s", url)
        return False

def check_server_status():
    """Polls the server for its current status and round."""
    url = f"{SERVER_URL}/status"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json() # Returns {"status": "...", "current_round": ...}
        logger.warning("Could not get server status: %s", response.status_code)
        return None
    except requests.exceptions.ConnectionError:
        logger.warning("Server connection failed during status check.")
        return None

# --- Configuration for Data ---
#TOTAL_CLIENTS = 3 
#BATCH_SIZE = 32
#DIRICHLET_ALPHA = 0.5  # <-- Controls Non-IID level. Lower = More non-IID
#RANDOM_SEED = 42       # <-- To ensure clients get consistent partitions

def get_client_dataloader(client_id_num):
    """
    Downloads CIFAR-10 and returns a DataLoader for a
    specific client's data partition (Non-IID via Dirichlet).
    """
    logger.info(f"Loading CIFAR-10 dataset for Non-IID partition...")
    
    # Set seeds to ensure all clients generate the same partition map
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    except Exception as e:
        logger.error(f"Failed to download CIFAR-10: {e}")
        raise
        
    # --- Non-IID Partitioning using Dirichlet ---
    num_classes = 10
    total_size = len(train_dataset)
    
    # Get all labels
    labels = np.array(train_dataset.targets)
    
    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # client_partitions will be a list of lists,
    # where each inner list contains the data indices for a client
    client_partitions = [[] for _ in range(config.TOTAL_CLIENTS)]

    # Generate the Dirichlet distribution for each class
    # This determines what proportion of each class each client gets
    for k in range(num_classes):
        img_idx_k = class_indices[k]
        np.random.shuffle(img_idx_k) # Shuffle indices for this class
        
        # Get a Dirichlet distribution sample
        proportions = np.random.dirichlet(np.repeat(config.DIRICHLET_ALPHA, config.TOTAL_CLIENTS))
        
        # Calculate the cumulative sum of proportions (to slice the data)
        proportions_cumsum = (np.cumsum(proportions) * len(img_idx_k)).astype(int)[:-1]
        
        # Split the class indices among clients
        split_indices = np.split(img_idx_k, proportions_cumsum)
        
        for i in range(config.TOTAL_CLIENTS):
            client_partitions[i].extend(split_indices[i])
            
    # --- End Partitioning ---

    # Get the indices for *this* client
    # (client_id_num is 1-based, so we subtract 1)
    client_indices = client_partitions[client_id_num - 1]
    
    # Create a subset for this client
    client_subset = Subset(train_dataset, client_indices)
    
    # Create the DataLoader
    client_loader = DataLoader(client_subset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=2)
    
    # Log the class distribution for this client
    class_counts = {i: 0 for i in range(num_classes)}
    for idx in client_indices:
        label = labels[idx]
        class_counts[label] += 1
    logger.info(f"Client {CLIENT_ID} loaded partition with {len(client_subset)} samples.")
    logger.info(f"Client {CLIENT_ID} class distribution: {class_counts}")

    return client_loader

# --- Main Client Loop ---

def main():
    logger.info("--- Starting FL Client %s ---", CLIENT_ID)
    
    # 1. Register with the server
    if not register_client():
        logger.error("Could not register with server. Exiting.")
        return

    # --- Instantiate your REAL model ---
    global_model = get_model()
    logger.info("Model %s instantiated.", global_model.__class__.__name__)
    
    # --- NEW: Load this client's data partition ---
    try:
        # Get the number from the CLIENT_ID (e.g., "client_001" -> 1)
        client_id_num = int(CLIENT_ID.split('_')[-1])
        if client_id_num > config.TOTAL_CLIENTS or client_id_num < 1:
            raise ValueError(f"CLIENT_ID {CLIENT_ID} not in expected range 1-{config.TOTAL_CLIENTS}")
        
        client_dataloader = get_client_dataloader(client_id_num)
    except Exception as e:
        logger.error(f"Failed to load data: {e}. Exiting.")
        return
    # --- END NEW ---
    
    current_round = 0

    # 2. Download initial model (Round 0)
    logger.info("Waiting for initial model (round 0)...")
    model_state = None
    while model_state is None:
        model_state = download_global_model(current_round)
        if model_state is None:
            time.sleep(5)

    # ===================================================================
    # Main Training Loop
    # ===================================================================
    while True:
        logger.info("--- Starting Round %d ---", current_round)
        
        # 3. Load model state and train
        try:
            # 1. Load the downloaded state into your model
            set_model_state(global_model, model_state)
            
            # 2. Train the model using the real data
            logger.info("Starting local training...")
            # --- NEW: Measure training time ---
            start_time = time.time()
            num_samples, peak_gpu_mb, peak_ram_mb = train_model(global_model, client_dataloader)
            end_time = time.time()

            training_time_sec = end_time - start_time
            logger.info(f"Local training complete in {training_time_sec:.2f}s.")
            # ---

            new_model_state = get_model_state(global_model)

        except Exception as e:
            logger.error(f"Local training for round {current_round} failed: {e}", exc_info=True)
            time.sleep(15) 
            continue
        
        
        # 4. Submit model update
        logger.info("Submitting trained model update from %d samples...", num_samples)
        # --- NEW: Create metrics dict ---
        client_metrics = {
           "training_time_sec": training_time_sec,
            "peak_gpu_mb": peak_gpu_mb,
            "peak_ram_mb": peak_ram_mb
            # payload_size_mb will be added by submit_model_update
        }
        if not submit_model_update(new_model_state, num_samples, client_metrics):
            logger.warning("Failed to submit update. Retrying in 10s.")
            time.sleep(10)
            continue 

        logger.info("Update submitted. Waiting for next round...")
        
        # 5. Wait for server to finish aggregation
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
                
                new_model_state_response = download_global_model(server_round)
                
                if new_model_state_response:
                    model_state = new_model_state_response 
                    current_round = server_round  
                    break 
                else:
                    logger.warning("Failed to download new model. Retrying...")

if __name__ == "__main__":
    main()