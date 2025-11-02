import torch
import torch.nn as nn
import torch.optim as optim
import logging
import config
import torch.cuda
import psutil
import os

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Configuration for Training ---
#LOCAL_EPOCHS = 3
#LEARNING_RATE = 0.01
#MOMENTUM = 0.9


def train_model(model, client_data):
    """
    Performs local training on the client's model.
    
    Returns:
        (int) num_samples: Number of samples trained on.
        (float) peak_gpu_mb: Peak GPU memory used in megabytes.
        (float) peak_ram_mb: Peak System RAM used in megabytes.
    """
    
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

    model.to(device)
    logger.info(f"Training on device: {device}")
    
    # --- GPU Memory Stats ---
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        
    # --- System RAM Stats ---
    process = psutil.Process(os.getpid())
    peak_ram_mb = 0.0
    # ---

    data_loader = client_data 
    num_samples = len(data_loader.dataset)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM)

    model.train() 

    for epoch in range(config.LOCAL_EPOCHS):
        running_loss = 0.0
        
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # --- NEW: Check RAM usage per batch ---
            current_ram_bytes = process.memory_info().rss
            peak_ram_mb = max(peak_ram_mb, current_ram_bytes / (1024 * 1024))
            # ---
        
        epoch_loss = running_loss / len(data_loader)
        logger.info(f"Epoch {epoch+1}/{config.LOCAL_EPOCHS} - Loss: {epoch_loss:.4f}")

    logger.info("Local training complete.")
    
    # --- Get peak GPU memory usage ---
    peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    peak_gpu_mb = peak_gpu_bytes / (1024 * 1024)
    # ---

    # Return all three metrics
    return num_samples, peak_gpu_mb, peak_ram_mb