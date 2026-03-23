import logging
from flask import Flask, request, jsonify, send_file
from collections import OrderedDict
import os
import io
import signal
import threading
from threading import Timer, Lock
import psutil
import time
import time as _time
import random
import json
import json as json_stdlib
import re

# --- ML Imports ---
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from shared.models import get_model
from server.strategies import get_strategy
import config

# --- Partial Training --- Import base class for isinstance checks
# HeteroFL, FedRolex, and FedPrune all inherit from this
from server.strategies.partial_training_base import PartialTrainingStrategy

app = Flask(__name__)

werkzeug_logger = logging.getLogger('werkzeug')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

LOG_DIR_TB = os.path.join(PROJECT_ROOT, "fl_logs", "tensorboard")
MODEL_DIR = os.path.join(PROJECT_ROOT, "fl_logs", "models")

os.makedirs(LOG_DIR_TB, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

tb_writer = SummaryWriter(log_dir=LOG_DIR_TB)

# 2. Define a custom filter
class StatusFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "GET /status" in msg or "POST /register" in msg:
            return False 
        return True

# 3. Add the filter to the logger
werkzeug_logger.addFilter(StatusFilter())

# --- Configuration ---
fl_state = {
    "status": "WAITING", 
    "current_round": 0,
    "client_updates": [],
    "registered_clients": set(),
    "aggregation_lock": Lock(), 
    "round_timer": None,
    "fedprune_round0_ready": False,  # Track if round 0 indices are computed
}

# Structured communication metrics — written to disk every round
COMM_METRICS_PATH = os.path.join("fl_logs", "round_metrics.jsonl")

def log_round_metrics(round_num, metrics_dict):
    """Write one line of JSON to round_metrics.jsonl for post-experiment parsing."""
    metrics_dict["round"] = round_num
    metrics_dict["wall_timestamp"] = time.time()
    try:
        with open(COMM_METRICS_PATH, "a") as f:
            f.write(json_stdlib.dumps(metrics_dict) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write round metrics: {e}")


# --- Global strategy variable ---
fl_strategy = None
# ---

def get_model_path(round_num):
    """Returns the file path for a specific round's model."""
    return os.path.join(MODEL_DIR, f"model_round_{round_num}.pth")

# --- PARTIAL TRAINING --- Per-client model path
def get_client_model_path(round_num, client_id):
    """Returns a per-client model path (used by FedPrune for unique indices)."""
    return os.path.join(MODEL_DIR, f"model_round_{round_num}_{client_id}.pth")


def cleanup_model_dir():
    """Remove ALL .pth files from model directory. Called at startup."""
    removed = 0
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.pth'):
            try:
                os.remove(os.path.join(MODEL_DIR, f))
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Cleaned up {removed} model files from previous run(s).")


def cleanup_old_round_files(keep_round):
    """
    Remove model files from rounds older than keep_round.
    Keeps only the current round's global + per-client files.
    Called after each successful aggregation.
    """
    removed = 0
    for f in os.listdir(MODEL_DIR):
        if not f.endswith('.pth') or f == config.SAVED_MODEL_NAME:
            continue
        # Extract round number from filename: model_round_N.pth or model_round_N_client_XXX.pth
        m = re.match(r'model_round_(\d+)', f)
        if m:
            file_round = int(m.group(1))
            if file_round < keep_round:
                try:
                    os.remove(os.path.join(MODEL_DIR, f))
                    removed += 1
                except OSError:
                    pass
    if removed > 0:
        logger.info(f"Cleaned up {removed} model files from rounds < {keep_round}.")


def setup_initial_model():
    """
    Creates the very first model for round 0 and saves it to disk.
    Initializes the aggregation strategy.
    """
    global fl_strategy
    
    # Clean up any leftover model files from previous runs
    cleanup_model_dir()
    
    # 1. Create the model instance first
    initial_model = get_model(config.MODEL_NAME)
    
    # 2. Initialize Strategy
    fl_strategy = get_strategy(
        config.AGGREGATION_STRATEGY, 
        global_model=initial_model,
        lr=config.SERVER_LEARNING_RATE,
        momentum=config.SERVER_MOMENTUM,
        # FedPrune-specific (ignored by other strategies via **kwargs)
        ema_decay=config.EMA_DECAY,
        importance_alpha=config.IMPORTANCE_ALPHA,
        total_rounds=config.TOTAL_ROUNDS,
    )
    
    # 3. Save the state_dict (tensors) directly to DISK
    save_path = get_model_path(0)
    torch.save(initial_model.state_dict(), save_path)
    
    logger.info(f"Initial model for round 0 created and saved to {save_path}.")
    logger.info(f"Strategy: {config.AGGREGATION_STRATEGY} initialized.")

# --- Helper Functions ---

def load_test_data():
    """Loads the test dataset (CIFAR-10 or CIFAR-100 based on config)."""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    dataset_cls = torchvision.datasets.CIFAR100 if config.DATASET_NAME == "CIFAR100" \
                  else torchvision.datasets.CIFAR10
    test_dataset = dataset_cls(
        root='./data', train=False, download=False, transform=transform)
    logger.info(f"Loaded test dataset: {config.DATASET_NAME} ({len(test_dataset)} samples)")
    return DataLoader(test_dataset, batch_size=256, shuffle=False)


def evaluate_model(model_state_dict, test_loader):
    """Evaluates a model state_dict on the test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config.MODEL_NAME)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    # Per-class tracking — detect num_classes from model's final layer
    n_classes = 100 if config.DATASET_NAME == "CIFAR100" else 10
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Per-class
            for c in range(n_classes):
                mask = target == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (predicted[mask] == c).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)
    
    # Log per-class accuracy
    class_accs = []
    for c in range(n_classes):
        if class_total[c] > 0:
            ca = 100. * class_correct[c] / class_total[c]
            class_accs.append(ca)
        else:
            class_accs.append(0.0)
    
    class_acc_str = ",".join(f"{a:.1f}" for a in class_accs)
    class_std = float(np.std(class_accs))
    logger.info(f"PER_CLASS_ACC: [{class_acc_str}] std={class_std:.2f}")
    
    return accuracy, avg_loss


# --- PARTIAL TRAINING --- Compute and save per-client models
def prepare_fedprune_round(round_num):
    """
    For Partial Training: compute indices for all registered clients and save
    per-client composite payloads to disk.
    """
    if not isinstance(fl_strategy, PartialTrainingStrategy):
        return
    
    participating = list(fl_state["registered_clients"])
    if not participating:
        logger.warning(f"{fl_strategy.METHOD_NAME}: No registered clients to compute indices for.")
        return
    
    logger.info(f"{fl_strategy.METHOD_NAME}: Computing indices for {len(participating)} clients (round {round_num})")
    fl_strategy.compute_client_indices(participating)
    
    # Load the shared model weights
    model_path = get_model_path(round_num)
    if not os.path.exists(model_path):
        logger.error(f"{fl_strategy.METHOD_NAME}: Model file {model_path} not found!")
        return
    
    model_state = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Save a per-client composite payload
    for client_id in participating:
        payload = fl_strategy.get_payload_for_client(client_id, model_state)
        client_path = get_client_model_path(round_num, client_id)
        torch.save(payload, client_path)
    
    logger.info(f"{fl_strategy.METHOD_NAME}: Saved {len(participating)} per-client payloads for round {round_num}")


def check_and_aggregate(test_loader):
    """
    Performs aggregation, evaluation, model save, and round advancement.
    Note: fl_state["status"] is already set to "AGGREGATING" by the caller.
    """
    # 1. Cancel Timer if running
    if fl_state["round_timer"]:
        fl_state["round_timer"].cancel()
        fl_state["round_timer"] = None
        
    current_round = fl_state["current_round"]
    new_global_model_tensors = None
    
    if len(fl_state["client_updates"]) == 0:
        logger.warning(f"No updates received for round {current_round}. Skipping aggregation.")
    else:
        logger.info(f"--- Aggregating {len(fl_state['client_updates'])} updates for round {current_round} ---")
        
        psutil.cpu_percent(interval=None)
        ram_before_mb = psutil.virtual_memory().used / (1024 * 1024)
        start_time = time.time()
        # 2. Aggregate
        t0 = _time.time()
        aggregation_result = fl_strategy.aggregate(fl_state["client_updates"])
        t1 = _time.time()
        logger.info(f"TIMING: AGGREGATE={t1 - t0:.2f}s")

        agg_duration = time.time() - start_time
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_after_mb = psutil.virtual_memory().used / (1024 * 1024)

        tb_writer.add_scalar("System/CPU_Usage_Percent", cpu_usage, current_round)
        tb_writer.add_scalar("System/RAM_Usage_MB", ram_after_mb, current_round)
        tb_writer.add_scalar("System/RAM_Spike_MB", ram_after_mb - ram_before_mb, current_round)
        tb_writer.add_scalar("System/Aggregation_Time_Sec", agg_duration, current_round)
        tb_writer.flush()

        logger.info(f"PROFILING: CPU {cpu_usage}%, RAM {ram_after_mb:.2f}MB, Time {agg_duration:.2f}s")
        
        # 3. Unwrap Payload (Handle Scaffold/FedPrune vs Standard)
        if isinstance(aggregation_result, dict) and "model_state" in aggregation_result:
            new_global_model_tensors = aggregation_result["model_state"]
            payload_to_save = aggregation_result 
        else:
            new_global_model_tensors = aggregation_result
            payload_to_save = aggregation_result

        # 4. Evaluate
        if new_global_model_tensors:
            t2 = _time.time()
            accuracy, loss = evaluate_model(new_global_model_tensors, test_loader)
            t3 = _time.time()
            logger.info(f"TIMING: EVALUATE={t3 - t2:.2f}s")
            tb_writer.add_scalar("Global/Accuracy", accuracy, current_round)
            tb_writer.add_scalar("Global/Loss", loss, current_round)
            tb_writer.flush()
            
            logger.info(f"--- Round {current_round} Complete. Accuracy: {accuracy:.2f}%, Loss: {loss:.4f} ---")

            # Log structured round metrics for communication analysis
            client_comms = []
            for update in fl_state["client_updates"]:
                cid = update["client_id"]
                m = update.get("metrics", {})
                client_comms.append({
                    "client_id": cid,
                    "upload_mb": m.get("payload_size_mb", 0),
                    "training_time_sec": m.get("training_time_sec", 0),
                    "peak_gpu_mb": m.get("peak_gpu_mb", 0),
                    "peak_ram_mb": m.get("peak_ram_mb", 0),
                })

            log_round_metrics(current_round, {
                "n_clients": len(fl_state["client_updates"]),
                "accuracy": accuracy,
                "loss": loss,
                "agg_duration_sec": agg_duration,
                "clients": client_comms,
            })
            
            # --- PARTIAL TRAINING --- Log EMA diagnostics to TensorBoard
            if isinstance(fl_strategy, PartialTrainingStrategy):
                caps = [fl_strategy._get_capacity(c) for c in fl_state["registered_clients"]]
                cap_std = float(np.std(caps)) if caps else 0.0
                tb_writer.add_scalar("PartialTraining/cap_std", cap_std, current_round)
                # imp_frac is FedPrune-specific (capacity-adaptive extraction)
                if hasattr(fl_strategy, '_compute_imp_frac'):
                    imp_frac = min(0.9, max(0.5, 0.5 + cap_std))
                    tb_writer.add_scalar("PartialTraining/imp_frac", imp_frac, current_round)
            
            # 5. Save new model
            next_round = current_round + 1
            
            # For Partial Training: save only model_state (per-client payloads built separately)
            if isinstance(fl_strategy, PartialTrainingStrategy):
                t4 = _time.time()
                save_path = get_model_path(next_round)
                torch.save(new_global_model_tensors, save_path)
                t5 = _time.time()
                logger.info(f"TIMING: SAVE_MODEL={t5 - t4:.2f}s")
            else:
                save_path = get_model_path(next_round)
                torch.save(payload_to_save, save_path)
            
            logger.info(f"Saved new model to {save_path}")
            
            # Clean up old round files to prevent disk bloat
            cleanup_old_round_files(keep_round=next_round)
        else:
            next_round = current_round + 1
            logger.warning("Aggregation returned None. Reusing previous model.")

    # 6. Check Termination
    next_round = current_round + 1
    if next_round >= config.TOTAL_ROUNDS:
        fl_state["status"] = "TRAINING_COMPLETE"
        logger.info(f"--- All {config.TOTAL_ROUNDS} rounds complete! ---")
        
        final_path = os.path.join(MODEL_DIR, config.SAVED_MODEL_NAME)
        if new_global_model_tensors:
            torch.save(new_global_model_tensors, final_path)
            logger.info(f"Final model saved to {final_path}")
        
        Timer(5, shutdown_server).start()
        return
    
    # 7. Advance to next round
    fl_state["current_round"] = next_round
    
    # --- PARTIAL TRAINING --- Compute indices for next round (all clients are known by now)
    t6 = _time.time()
    prepare_fedprune_round(next_round)
    t7 = _time.time()
    logger.info(f"TIMING: PREPARE_ROUND={t7 - t6:.2f}s")

    logger.info(f"Total round processing time: {t7 - t0:.2f}s")
    
    start_next_round_timer(test_loader)


def trigger_aggregation(test_loader):
    """Called when the round timer expires."""
    with fl_state["aggregation_lock"]:
        if fl_state["status"] == "WAITING":
            logger.info("Round timeout reached. Triggering aggregation.")
            fl_state["status"] = "AGGREGATING"
            check_and_aggregate(test_loader)


def start_next_round_timer(test_loader):
    global fl_state
    
    round_num = fl_state["current_round"]
    logger.info(f"--- Starting Round {round_num}. Timeout: {config.ROUND_TIMEOUT_SEC}s ---")
    fl_state["status"] = "WAITING"
    fl_state["client_updates"] = []
    
    fl_state["round_timer"] = Timer(
        config.ROUND_TIMEOUT_SEC, 
        trigger_aggregation, 
        args=[test_loader] 
    )
    fl_state["round_timer"].start()

def shutdown_server():
    logger.info("--- Training complete. Sending shutdown signal. ---")
    os.kill(os.getpid(), signal.SIGINT)

# --- Flask Routes ---

logging.basicConfig(level=logging.INFO, format='INFO:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    client_id = data.get("client_id")
    capacity = data.get("capacity", 0.5)  # Clients report their capacity
    
    with fl_state["aggregation_lock"]:
        if client_id not in fl_state["registered_clients"]:
            fl_state["registered_clients"].add(client_id)
            
            # --- PARTIAL TRAINING --- Register this client's capacity dynamically
            if isinstance(fl_strategy, PartialTrainingStrategy):
                fl_strategy.register_capacity(client_id, capacity)
            
            logger.info(
                f"Client {client_id} registered (capacity={capacity:.2f}). "
                f"Total: {len(fl_state['registered_clients'])}/{config.TOTAL_CLIENTS}"
            )
        
        # Start round timer on first registration (unchanged)
        if fl_state["current_round"] == 0 and fl_state["round_timer"] is None:
            logger.info("First client registered. Starting Round 0 timer.")
            with app.app_context():
                test_loader = app.config['TEST_LOADER'] 
                start_next_round_timer(test_loader)    
        
        # --- PARTIAL TRAINING --- Once ALL expected clients have registered,
        # compute round 0 indices so everyone gets pre-computed payloads.
        if (isinstance(fl_strategy, PartialTrainingStrategy) and 
            not fl_state["fedprune_round0_ready"] and
            len(fl_state["registered_clients"]) >= config.TOTAL_CLIENTS):
            
            logger.info("All clients registered. Computing partial-training round 0 indices.")
            prepare_fedprune_round(0)
            fl_state["fedprune_round0_ready"] = True
        
    return jsonify({"status": fl_state["status"]})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": fl_state["status"],
        "current_round": fl_state["current_round"]
    })

@app.route('/download_model', methods=['GET'])
def download_model():
    try:
        requested_round = request.args.get('round', type=int)
        if requested_round is None:
            return jsonify({"error": "'round' parameter is required"}), 400
        
        # --- PARTIAL TRAINING --- Serve per-client payload
        if isinstance(fl_strategy, PartialTrainingStrategy):
            client_id = request.args.get('client_id', type=str)
            
            if client_id:
                # Try pre-computed file first
                client_path = get_client_model_path(requested_round, client_id)
                if os.path.exists(client_path):
                    dl_bytes = os.path.getsize(client_path)                         # <<< NEW
                    logger.info(
                        f"COMM_DL: client={client_id} round={requested_round} "     # <<< NEW
                        f"bytes={dl_bytes} timestamp={time.time():.4f}"             # <<< NEW
                    )                                                                # <<< NEW
                    logger.info(f"Serving per-client model for {client_id} from {client_path}")
                    return send_file(
                        client_path,
                        mimetype='application/octet-stream',
                        as_attachment=True,
                        download_name=f'model_round_{requested_round}.pth'
                    )
                else:
                    # Pre-computed file not ready — generate on the fly
                    logger.warning(
                        f"Per-client model not found for {client_id} round {requested_round}. "
                        f"Generating on-the-fly."
                    )
                    base_path = get_model_path(requested_round)
                    if os.path.exists(base_path):
                        model_state = torch.load(base_path, map_location='cpu', weights_only=True)
                        payload = fl_strategy.get_payload_for_client(client_id, model_state)
                        torch.save(payload, client_path)
                        dl_bytes = os.path.getsize(client_path)                     # <<< NEW
                        logger.info(
                            f"COMM_DL: client={client_id} round={requested_round} " # <<< NEW
                            f"bytes={dl_bytes} timestamp={time.time():.4f}"         # <<< NEW
                        )                                                            # <<< NEW
                        return send_file(
                            client_path,
                            mimetype='application/octet-stream',
                            as_attachment=True,
                            download_name=f'model_round_{requested_round}.pth'
                        )
            
            logger.warning("Partial-training active but no client_id in download request. Serving base model.")
        
        # Standard path: serve shared model file
        file_path = get_model_path(requested_round)
        
        if os.path.exists(file_path):
            dl_bytes = os.path.getsize(file_path)                                   # <<< NEW
            logger.info(
                f"COMM_DL: client=shared round={requested_round} "                  # <<< NEW
                f"bytes={dl_bytes} timestamp={time.time():.4f}"                     # <<< NEW
            )                                                                        # <<< NEW
            logger.info(f"Serving model from {file_path}")
            return send_file(
                file_path,
                mimetype='application/octet-stream',
                as_attachment=True,
                download_name=f'model_round_{requested_round}.pth'
            )
        else:
            logger.warning(f"Client requested model for round {requested_round}, not found at {file_path}.")
            return jsonify({"error": f"Model for round {requested_round} not found."}), 404
            
    except Exception as e:
        logger.error(f"Error in /download_model: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/submit_update', methods=['POST'])
def submit_update():
    if fl_state["status"] != "WAITING":
        return jsonify({"error": "Server is not accepting updates right now."}), 400
    
    if 'model' not in request.files or 'json' not in request.form:
        return jsonify({"error": "Missing model or metadata"}), 400

    try:
        # 1. Parse JSON Metadata
        metadata = json.loads(request.form['json'])
        client_id = metadata.get('client_id')
        num_samples = metadata.get('num_samples')
        metrics = metadata.get('metrics', {})
        
        if not all([client_id, num_samples is not None]):
            return jsonify({"error": "Missing required metadata"}), 400
        
        # Client Dropout Simulation
        if config.CLIENT_DROPOUT_RATE > 0 and random.random() < config.CLIENT_DROPOUT_RATE:
            logger.warning(f"SIMULATING DROPOUT: Ignoring update from {client_id}.")
            return jsonify({"status": "Update received"})
        
        # 2. Read binary data (ONCE)
        file_bytes = request.files['model'].read()
        
        # 3. Log upload size
        logger.info(
            f"COMM_UL: client={client_id} round={fl_state['current_round']} "
            f"bytes={len(file_bytes)} timestamp={time.time():.4f}"
        )
        
        # 4. Deserialize
        binary_data = torch.load(io.BytesIO(file_bytes), map_location='cpu', weights_only=True)
        
        # 5. Unwrap based on payload structure
        if isinstance(binary_data, dict) and "model_state" in binary_data:
            client_state_dict = binary_data["model_state"]
            if "tensor_metrics" in binary_data:
                metrics.update(binary_data["tensor_metrics"])
        else:
            client_state_dict = binary_data
        
        # 6. Check JSON metadata for compact submodel flag
        is_submodel = metadata.get("is_submodel", False)
        if is_submodel:
            client_state_dict = {
                "model_state": client_state_dict,
                "is_submodel": True,
            }
        
    except Exception as e:
        logger.error(f"Error parsing client update: {e}", exc_info=True)
        return jsonify({"error": "Failed to parse update"}), 400
    
    with fl_state["aggregation_lock"]:
        if fl_state["status"] != "WAITING":
            return jsonify({"error": "Server is not accepting updates right now (locked)."}), 400
        
        for update in fl_state["client_updates"]:
            if update["client_id"] == client_id:
                logger.warning(f"Client {client_id} tried to submit a second update.")
                return jsonify({"error": "Already received update for this round"}), 400
        
        # Store update
        fl_state["client_updates"].append({
            "client_id": client_id,
            "num_samples": num_samples,
            "model_update": client_state_dict,
            "metrics": metrics,
            "is_submodel": is_submodel,
        })
        
        logger.info(
            f"Update from {client_id} (Round {fl_state['current_round']}): "
            f"Upload: {metrics.get('payload_size_mb', 0):.2f}MB "
            f"({len(fl_state['client_updates'])}/{config.MIN_CLIENTS_PER_ROUND})"
        )
        
        # Non-Blocking Aggregation Trigger
        if len(fl_state["client_updates"]) >= config.MIN_CLIENTS_FOR_AGGREGATION:
            logger.info(f"Quorum met. Triggering aggregation in BACKGROUND.")
            fl_state["status"] = "AGGREGATING"
            
            agg_thread = threading.Thread(
                target=check_and_aggregate,
                args=[app.config['TEST_LOADER']]
            )
            agg_thread.start()
        
    return jsonify({"status": "Update received"})

if __name__ == '__main__':
    try:
        test_loader = load_test_data()
        app.config['TEST_LOADER'] = test_loader
    except Exception as e:
        logger.error(f"Failed to load test data: {e}. Exiting.")
        exit(1)
        
    setup_initial_model() 
    
    logger.info(f"--- FL Server starting... ---")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)