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
from client.trainer import train_model 
from client.model_utils import get_model_bytes, set_model_from_bytes, prepare_upload_payload
from client.algorithms import get_client_algorithm
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
    """Registers the client with the server, reporting its capacity."""
    url = f"{SERVER_URL}/register"
    try:
        response = requests.post(url, json={
            "client_id": CLIENT_ID,
            "capacity": config.CLIENT_CAPACITY,  # Report capacity to server
        })
        if response.status_code == 200:
            logger.info(
                "Successfully registered (capacity=%.2f). Server status: %s",
                config.CLIENT_CAPACITY,
                response.json().get("status")
            )
            return True
        elif response.status_code == 400: # Already registered
             logger.info("Client already registered. Proceeding.")
             return True
        logger.error("Failed to register. Status: %s, Body: %s", response.status_code, response.text)
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to server at %s", url)
        return False

def download_global_model(round_number, max_retries=5):
    """Downloads the global model with transfer timing instrumentation."""
    url = f"{SERVER_URL}/download_model"
    
    if config.NETWORK_LATENCY_RATE > 0 and random.random() < config.NETWORK_LATENCY_RATE:
        delay = config.NETWORK_LATENCY_DELAY_SEC
        logger.warning(f"SIMULATING NETWORK LATENCY: Delaying download by {delay}s...")
        time.sleep(delay)
    
    for attempt in range(max_retries):
        try:
            dl_start = time.time()
            response = requests.get(url, params={
                "round": round_number,
                "client_id": CLIENT_ID
            }, timeout=120)
            dl_end = time.time()
            
            if response.status_code == 200:
                dl_bytes = len(response.content)
                dl_time = dl_end - dl_start
                throughput = dl_bytes * 8 / 1024 / max(dl_time, 0.001)
                
                logger.info(
                    f"Downloaded model for round {round_number} ({dl_bytes} bytes)."
                )
                # Structured communication metric
                logger.info(
                    f"COMM_DL_CLIENT: round={round_number} "
                    f"bytes={dl_bytes} "
                    f"transfer_sec={dl_time:.4f} "
                    f"throughput_kbps={throughput:.1f}"
                )
                return response.content
                
            logger.error(f"Failed download. Status: {response.status_code}")
            
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            wait = 2 ** attempt + random.random()
            logger.warning(
                f"Download failed (attempt {attempt+1}/{max_retries}): "
                f"{type(e).__name__}. Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
    
    logger.error(f"Failed to download model after {max_retries} attempts.")
    return None

def submit_model_update(model, num_samples, metrics, neuron_indices=None):
    """
    Submits the locally trained model update to the server.
    
    If neuron_indices is provided and the method supports compact uploads,
    only the trained neurons are serialized and sent (smaller uplink).
    
    Args:
        model: Trained model
        num_samples: Number of training samples used
        metrics: Dict of training metrics (may include tensor_metrics like
                 FIARSE importance scores)
        neuron_indices: Dict {layer_name: [indices]} for compact upload,
                       or None for full upload
    """
    url = f"{SERVER_URL}/submit_update"
    
    try:
        # 1. Prepare Binary Data — compact or full depending on method
        binary_data = model.state_dict()
        
        # Check for Tensor metrics (SCAFFOLD delta_c, FIARSE importance)
        tensor_metrics = {}
        keys_to_remove = []
        
        for key, value in metrics.items():
            if hasattr(value, 'cpu') or (isinstance(value, dict) and len(value) > 0 
                    and hasattr(next(iter(value.values()), None), 'cpu')):
                tensor_metrics[key] = value
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del metrics[k]

        # Decide upload format
        if neuron_indices and config.CLIENT_ALGO not in ("FIARSE",):
            from shared.submodel_utils import extract_trained_submodel
            submodel_state = extract_trained_submodel(model, neuron_indices)
            
            # Send flat tensor dict — no wrapper.
            # The is_submodel flag travels in the JSON metadata.
            binary_data = submodel_state
            
            # tensor_metrics (e.g. FIARSE importance) won't apply here
            # since we're in the non-FIARSE branch, but handle just in case
            if tensor_metrics:
                binary_data = {
                    "model_state": submodel_state,
                    "tensor_metrics": tensor_metrics,
                }
        else:
            # Full upload (standard, FIARSE, or no indices)
            if tensor_metrics:
                binary_data = {
                    "model_state": binary_data,
                    "tensor_metrics": tensor_metrics,
                }
            
        # Serialize
        model_bytes = get_model_bytes(binary_data)
        
        # 2. Update Upload Size Metric
        metrics["payload_size_mb"] = len(model_bytes) / (1024 * 1024)
        
        # 3. Prepare Metadata (Safe JSON)
        metadata = {
            "client_id": CLIENT_ID,
            "num_samples": num_samples,
            "metrics": metrics,
            "is_submodel": True if (neuron_indices and config.CLIENT_ALGO not in ("FIARSE",)) else False,
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

        ul_start = time.time()
        response = requests.post(url, files=files)
        ul_end = time.time()
        
        if response.status_code == 200:
            ul_time = ul_end - ul_start
            throughput = len(model_bytes) * 8 / 1024 / max(ul_time, 0.001)
            
            logger.info("Successfully submitted model update.")
            # Structured communication metric
            logger.info(
                f"COMM_UL_CLIENT: "
                f"bytes={len(model_bytes)} "
                f"transfer_sec={ul_time:.4f} "
                f"throughput_kbps={throughput:.1f}"
            )
            return True
        
        logger.error(f"Failed to submit update. Status: {response.status_code}, Body: {response.text}")
        return False

    except Exception as e:
        logger.error(f"Error during model submission: {e}", exc_info=True)
        return False



def _extract_upload_submodel(model, indices):
    """Extract only the neurons this client trained for upload."""
    state = model.state_dict()
    submodel = {}
    
    layer_order = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and param.dim() >= 2:
            if name != 'fc3.weight':
                layer_order.append(name)
    
    prev_indices = None
    
    for layer_name in layer_order:
        if layer_name not in indices:
            continue
        
        out_idx = indices[layer_name]
        weight_key = layer_name
        bias_key = layer_name.replace('.weight', '.bias')
        
        w = state[weight_key]
        w_sub = w[out_idx]
        
        if prev_indices is not None and w_sub.dim() >= 2:
            if w_sub.dim() == 4:
                w_sub = w_sub[:, prev_indices]
            elif w_sub.dim() == 2:
                w_sub = w_sub[:, prev_indices]
        
        submodel[weight_key] = w_sub
        
        if bias_key in state:
            submodel[bias_key] = state[bias_key][out_idx]
        
        bn_prefix = layer_name.replace('conv', 'bn').replace('.weight', '')
        for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
            bn_key = bn_prefix + suffix
            if bn_key in state:
                submodel[bn_key] = state[bn_key][out_idx]
        
        prev_indices = out_idx
    
    fc3_w_key = 'fc3.weight'
    fc3_b_key = 'fc3.bias'
    if fc3_w_key in state:
        fc3_w = state[fc3_w_key]
        if prev_indices is not None:
            fc3_w = fc3_w[:, prev_indices]
        submodel[fc3_w_key] = fc3_w
    if fc3_b_key in state:
        submodel[fc3_b_key] = state[fc3_b_key]
    
    return {"model_state": submodel, "is_submodel": True, "indices": indices}

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
    Loads dataset (CIFAR-10 or CIFAR-100) and returns a DataLoader
    using pre-calculated indices from partitions.json.
    """
    logger.info(f"Loading data for client {client_id_num}...")

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    try:
        dataset_cls = torchvision.datasets.CIFAR100 if config.DATASET_NAME == "CIFAR100" \
                      else torchvision.datasets.CIFAR10
        train_dataset = dataset_cls(
            root='./data', train=True, download=False, transform=transform
        )
    except RuntimeError:
        logger.error(f"{config.DATASET_NAME} dataset not found in ./data! Did you run prepare_data.py?")
        raise

    partition_file = './data/partitions.json'
    if not os.path.exists(partition_file):
        raise FileNotFoundError(f"Partition file {partition_file} not found.")
    
    with open(partition_file, 'r') as f:
        partitions = json.load(f)
    
    client_indices = partitions.get(str(client_id_num))
    if client_indices is None:
        raise ValueError(f"No partition found for client {client_id_num}")

    client_subset = Subset(train_dataset, client_indices)
    
    client_loader = DataLoader(client_subset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=2)
    
    logger.info(f"Client {CLIENT_ID} loaded {len(client_subset)} samples from partitions.json.")
    return client_loader

# --- Main Client Loop ---

def main():
    logger.info("--- Starting FL Client %s (capacity=%.2f) ---", CLIENT_ID, config.CLIENT_CAPACITY)
    
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
            extra_payload = set_model_from_bytes(global_model, model_bytes)

            # 4. Train the model using the algorithm
            logger.info("Starting local training...")
            start_time = time.time()
            
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
        
        client_metrics = {
            "training_time_sec": training_time_sec,
            "peak_gpu_mb": peak_gpu_mb,
            "peak_ram_mb": peak_ram_mb,
            **training_metrics
        }
        
        if not submit_model_update(global_model, num_samples, client_metrics, 
                                   neuron_indices=extra_payload):
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