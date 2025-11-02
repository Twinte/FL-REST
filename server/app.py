import logging
from flask import Flask, request, jsonify
from collections import OrderedDict
import os
import signal
import threading
from threading import Timer, Lock
import random

# --- ML Imports ---
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
from client.model import get_model
import config

app = Flask(__name__)

werkzeug_logger = logging.getLogger('werkzeug')

# 2. Define a custom filter
class StatusFilter(logging.Filter):
    def filter(self, record):
        # Get the log message
        msg = record.getMessage()
        # Check if it's a /status or /register request
        if "GET /status" in msg or "POST /register" in msg:
            return False # Don't log this message
        return True

# 3. Add the filter to the logger
werkzeug_logger.addFilter(StatusFilter())

# --- Configuration ---
#MIN_CLIENTS_PER_ROUND = 3
#TOTAL_ROUNDS = 5 # Total number of rounds to run

fl_state = {
    "status": "WAITING", # WAITING, AGGREGATING, TRAINING_COMPLETE
    "current_round": 0,
    "client_updates": [],
    "registered_clients": set(),
    "aggregation_lock": Lock(), 
    "round_timer": None          
}

# We need a place to store the model (as a JSON-safe dict)
global_models_by_round = {}

# --- NEW: PyTorch Helper Functions ---

def serialize_model_state(state_dict_tensors):
    """
    Converts a PyTorch state_dict (tensors) into a
    JSON-serializable dictionary (lists).
    """
    return {
        key: tensor.cpu().tolist() 
        for key, tensor in state_dict_tensors.items()
    }

def deserialize_model_state(state_dict_lists):
    """
    Converts a JSON-serializable dictionary (lists) back
    into a PyTorch state_dict (tensors).
    """
    new_state_dict = OrderedDict()
    for key, param_list in state_dict_lists.items():
        # Load tensors onto CPU. The client will move to GPU if needed.
        new_state_dict[key] = torch.tensor(param_list, device='cpu')
    return new_state_dict

# --- UPDATED: Helper Functions ---

def load_test_data():
    """Loads the CIFAR-10 test dataset."""
    logger.info("Loading CIFAR-10 test dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=2)
    logger.info("Test dataset loaded.")
    return test_loader

def evaluate_model(model_state_tensors, test_loader):
    """Evaluates the global model on the test dataset."""
    
    # --- NEW: Check for GPU and set device ---
    # --- NEW: Set device based on config ---
    forced_device = config.DEVICE.lower()
    if forced_device == "cpu":
        device = torch.device("cpu")
    elif forced_device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available! Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else: # "auto" or any other value
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---

    logger.info(f"Evaluating on device: {device}")
    # --- END NEW ---
    
    model = get_model()
    model.load_state_dict(model_state_tensors)
    model.to(device) # <-- NEW: Move model to device
    model.eval() # Set model to evaluation mode
    
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculation
        for data in test_loader:
            images, labels = data
            
            # --- NEW: Move data and labels to the device ---
            images, labels = images.to(device), labels.to(device)
            # --- END NEW ---
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss

def setup_initial_model():
    """
    Creates the very first model for round 0, serializes it,
    and stores it.
    """
    global global_models_by_round
    
    # 1. Create an instance of the model
    initial_model = get_model()
    
    # 2. Get its state_dict (PyTorch Tensors)
    state_dict_tensors = initial_model.state_dict()
    
    # 3. Serialize it (Tensors -> Lists)
    serializable_state = serialize_model_state(state_dict_tensors)
    
    # 4. Save the JSON-safe version
    global_models_by_round[0] = serializable_state
    
    logger.info("Initial model for round 0 created and serialized.")

def aggregate_models(updates):
    """
    Performs Federated Averaging (FedAvg).
    
    :param updates: A list of client updates, where each update
                    contains the 'model_update' (serialized state)
                    and 'num_samples'.
    :return: A new global model state_dict (PyTorch Tensors).
    """
    logger.info(f"Aggregating {len(updates)} client updates...")
    
    # 1. Get total number of samples
    total_samples = sum(update['num_samples'] for update in updates)
    if total_samples == 0:
        logger.warning("No samples reported by clients. Aborting aggregation.")
        return None # Should not happen

    # 2. Create an empty state_dict for the new global model
    avg_state_dict = OrderedDict()

    # 3. Loop over all client updates
    for i, update in enumerate(updates):
        # 4. Deserialize client's model state (Lists -> Tensors)
        client_state_dict = deserialize_model_state(update['model_update'])
        
        # 5. Calculate client's weight
        client_weight = update['num_samples'] / total_samples
        
        # 6. Add its weighted parameters to the average
        for key in client_state_dict:
            weighted_param = client_state_dict[key] * client_weight
            
            if i == 0:
                # First client, just add the weighted param
                avg_state_dict[key] = weighted_param
            else:
                # Subsequent clients, add to the existing param
                avg_state_dict[key] += weighted_param
                        
    logger.info("Aggregation complete.")
    
    # 7. Return the new global model (as Tensors)
    return avg_state_dict

# file: server/app.py
def check_and_aggregate(test_loader):
    # This function is now called by EITHER the timer OR submit_update
    # It must be called from WITHIN the aggregation_lock
    
    # Check if aggregation is already happening (e.g., timer and submit raced)
    if fl_state["status"] == "AGGREGATING":
        return
        
    # 1. Cancel the timer (we are aggregating now)
    if fl_state["round_timer"]:
        fl_state["round_timer"].cancel()
        fl_state["round_timer"] = None
        
    # 2. Set status to prevent other calls
    fl_state["status"] = "AGGREGATING"
    current_round = fl_state["current_round"]
    
    # 3. Check if we have any updates to aggregate
    if len(fl_state["client_updates"]) == 0:
        logger.warning(f"No updates received for round {current_round}. Skipping aggregation.")
    else:
        logger.info(f"--- Aggregating {len(fl_state['client_updates'])} updates for round {current_round} ---")
        
        new_global_model_tensors = aggregate_models(fl_state["client_updates"])
        accuracy, loss = evaluate_model(new_global_model_tensors, test_loader)
        logger.info(f"--- Round {current_round} Complete. Accuracy: {accuracy:.2f}%, Loss: {loss:.4f} ---")
        new_global_model_serialized = serialize_model_state(new_global_model_tensors)
        global_models_by_round[current_round + 1] = new_global_model_serialized
        if current_round == config.TOTAL_ROUNDS - 1:
            try:
                save_path = os.path.join("/app/fl_logs", config.SAVED_MODEL_NAME)
                torch.save(new_global_model_tensors, save_path)
                logger.info(f"--- Final model saved to {save_path} ---")
            except Exception as e:
                logger.error(f"Failed to save final model: {e}", exc_info=True)

    # 4. Advance the round
    fl_state["current_round"] += 1
    
    # 5. Check for shutdown
    if fl_state["current_round"] >= config.TOTAL_ROUNDS:
        logger.info(f"--- All {config.TOTAL_ROUNDS} rounds complete. ---")
        fl_state["status"] = "TRAINING_COMPLETE"
        threading.Timer(2.0, shutdown_server).start()
    else:
        # 6. Start the next round's timer
        start_next_round_timer(test_loader)

def trigger_aggregation(test_loader):
    """Callback function for the round timer."""
    with fl_state["aggregation_lock"]:
        if fl_state["status"] == "WAITING": # Only trigger if still waiting
            logger.info(
                f"--- ROUND {fl_state['current_round']} TIMEOUT ---"
                f" Received {len(fl_state['client_updates'])}/{config.MIN_CLIENTS_PER_ROUND} updates."
            )
            check_and_aggregate(test_loader) # Force aggregation

def start_next_round_timer(test_loader): # <-- ADD test_loader AS ARGUMENT
    """Starts the timer for the current round."""
    global fl_state
    
    round_num = fl_state["current_round"]
    logger.info(f"--- Starting Round {round_num}. Timeout: {config.ROUND_TIMEOUT_SEC}s ---")
    fl_state["status"] = "WAITING"
    fl_state["client_updates"] = []
    
    # test_loader is now passed in, so we don't need app.config here
    
    # Start the new timer
    fl_state["round_timer"] = Timer(
        config.ROUND_TIMEOUT_SEC, 
        trigger_aggregation, 
        args=[test_loader] # Pass test_loader to the timer's callback
    )
    fl_state["round_timer"].start()

def shutdown_server():
    """Shuts down the server process."""
    logger.info("--- Training complete. Sending shutdown signal. ---")
    # This sends a SIGINT (Ctrl+C) to our own process
    os.kill(os.getpid(), signal.SIGINT)

# --- Flask Routes ---
# (These routes are IDENTICAL to the previous version,
#  but they now serve real serialized models)

logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    client_id = data.get("client_id")
    
    with fl_state["aggregation_lock"]:
        if client_id not in fl_state["registered_clients"]:
            fl_state["registered_clients"].add(client_id)
            logger.info(f"Client {client_id} registered. Total clients: {len(fl_state['registered_clients'])}")
        
        # --- MODIFIED: Pass test_loader to start the timer ---
        if fl_state["current_round"] == 0 and fl_state["round_timer"] is None:
            logger.info("First client registered. Starting Round 0 timer.")
            with app.app_context():
                test_loader = app.config['TEST_LOADER'] # Get test_loader here
                start_next_round_timer(test_loader)    # And pass it
        
    return jsonify({"status": fl_state["status"]})

@app.route('/status', methods=['GET'])
def get_status():
    """Returns the server's current status and round."""
    return jsonify({
        "status": fl_state["status"],
        "current_round": fl_state["current_round"]
    })

@app.route('/download_model', methods=['GET'])
def download_model():
    """Serves the global model for the requested round."""
    global global_models_by_round
    
    try:
        requested_round = request.args.get('round', type=int)
        if requested_round is None:
            return jsonify({"error": "'round' parameter is required"}), 400
            
        model_state = global_models_by_round.get(requested_round)
        
        if model_state:
            logger.info(f"Serving model for round {requested_round} to a client.")
            return jsonify({
                "model_state": model_state,
                "round": requested_round
            })
        else:
            logger.warning(f"Client requested model for round {requested_round}, which is not found.")
            return jsonify({"error": f"Model for round {requested_round} not found."}), 404
            
    except Exception as e:
        logger.error(f"Error in /download_model: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receives a model update from a client."""
    if fl_state["status"] != "WAITING":
        return jsonify({"error": "Server is not accepting updates right now."}), 400
    
    data = request.get_json()
    client_id = data.get("client_id")

    # --- NEW: Client Dropout Simulation ---
    if config.CLIENT_DROPOUT_RATE > 0 and random.random() < config.CLIENT_DROPOUT_RATE:
        logger.warning(
            f"SIMULATING DROPOUT: Ignoring update from {client_id} "
            f"for round {fl_state['current_round']}."
        )
        # We lie to the client, telling it the update was received
        # This simulates a packet being lost, or the client dying
        return jsonify({"status": "Update received"})
    # ---
    
    # Thread-safe check for updates
    with fl_state["aggregation_lock"]:
        # Re-check status inside lock
        if fl_state["status"] != "WAITING":
            return jsonify({"error": "Server is not accepting updates right now (locked)."}), 400
        
        # This 'model_update' is now a large JSON-safe dict of lists
        model_update = data.get("model_update")
        num_samples = data.get("num_samples")
        metrics = data.get("metrics", {}) # Get metrics dict, default to empty
        
        if not all([client_id, model_update, num_samples is not None]):
            return jsonify({"error": "Missing data in update"}), 400
        
        # Check for duplicates
        for update in fl_state["client_updates"]:
            if update["client_id"] == client_id:
                logger.warning(f"Client {client_id} tried to submit a second update for round {fl_state['current_round']}.")
                return jsonify({"error": "Already received update for this round"}), 400
        
        # Add update to the list
        fl_state["client_updates"].append(data)
        
        # Log the new metrics
        logger.info(
            f"Update from {client_id} (Round {fl_state['current_round']}): "
            f"Time: {metrics.get('training_time_sec', '?'):.1f}s, "
            f"RAM: {metrics.get('peak_ram_mb', '?'):.1f}MB, "
            f"GPU: {metrics.get('peak_gpu_mb', '?'):.1f}MB, "
            f"Upload: {metrics.get('payload_size_mb', '?'):.2f}MB "
            f"({len(fl_state['client_updates'])}/{config.MIN_CLIENTS_PER_ROUND})"
        )
        
        # --- MODIFIED: Check for "quorum" aggregation ---
        if len(fl_state["client_updates"]) >= config.MIN_CLIENTS_FOR_AGGREGATION:
            logger.info(f"Quorum of {config.MIN_CLIENTS_FOR_AGGREGATION} clients met. Triggering aggregation early.")
            check_and_aggregate(app.config['TEST_LOADER'])
        
    return jsonify({"status": "Update received"})

if __name__ == '__main__':
    # Load the test data once when the server starts
    try:
        test_loader = load_test_data()
        app.config['TEST_LOADER'] = test_loader
    except Exception as e:
        logger.error(f"Failed to load test data: {e}. Exiting.")
        exit(1)
        
    # Initialize the first model when the server starts
    setup_initial_model() 
    
    logger.info(f"--- FL Server starting... ---")
    logger.info(f"Waiting for {config.MIN_CLIENTS_PER_ROUND} clients per round.") # <-- Use config
    logger.info(f"Total rounds to run: {config.TOTAL_ROUNDS}") # <-- Use config
    
    app.run(host='0.0.0.0', port=5000, debug=False)