import torch
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

def get_model_state(model):
    """
    Gets the model's state_dict and converts tensors to lists
    for JSON serialization.
    """
    logger.debug("Serializing model state to JSON-safe format...")
    state_dict = model.state_dict()
    
    # Move tensors to CPU and convert to lists
    serializable_state = {
        key: tensor.cpu().tolist() 
        for key, tensor in state_dict.items()
    }
    return serializable_state

def set_model_state(model, model_state):
    """
    Converts a JSON-safe model state (lists) back into a 
    PyTorch state_dict (tensors) and loads it into the model.
    """
    logger.debug("Deserializing model state from JSON...")
    
    # Get the model's current state to match dtypes and devices
    current_state_dict = model.state_dict()
    
    # Create a new state_dict with the correct types
    new_state_dict = OrderedDict()
    for key, param_list in model_state.items():
        if key not in current_state_dict:
            logger.warning("Skipping key %s (not in model's state_dict)", key)
            continue
            
        current_tensor = current_state_dict[key]
        
        # Create tensor from list, matching original's dtype and device
        new_state_dict[key] = torch.tensor(param_list, 
                                           dtype=current_tensor.dtype,
                                           device=current_tensor.device)
    
    try:
        model.load_state_dict(new_state_dict)
        logger.debug("Successfully loaded new model state.")
    except Exception as e:
        logger.error("Failed to load state_dict: %s", e)
        # This can happen if model architectures don't match
        raise