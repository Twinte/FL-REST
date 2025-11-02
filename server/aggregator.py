import torch
import logging
import copy

logger = logging.getLogger(__name__)

def federated_average(client_updates):
    """
    Performs the Federated Averaging (FedAvg) algorithm.
    
    Args:
        client_updates (dict): A dictionary where keys are client_ids and
                               values are dicts {'state_dict': ..., 'num_samples': ...}
    
    Returns:
        dict: The new, aggregated global model state_dict.
    """
    if not client_updates:
        logger.warning("No client updates to aggregate.")
        return None

    logger.info(f"Starting FedAvg aggregation for {len(client_updates)} clients.")

    # 1. Calculate the total number of samples
    total_samples = sum(data['num_samples'] for data in client_updates.values())
    if total_samples == 0:
        logger.warning("No samples reported by clients. Cannot aggregate.")
        return None

    # 2. Get the keys from the first model to initialize a new one
    first_update = next(iter(client_updates.values()))['state_dict']
    
    # 3. Create a new state_dict, initialized with zeros
    # We must use copy.deepcopy to avoid modifying the original tensors
    new_global_state = {key: torch.zeros_like(tensor) for key, tensor in first_update.items()}
    
    # 4. Perform the weighted average
    for client_id, data in client_updates.items():
        weight = data['num_samples'] / total_samples
        state_dict = data['state_dict']
        
        for key in new_global_state:
            if key in state_dict:
                # new_state = new_state + (client_state * weight)
                new_global_state[key] += state_dict[key] * weight
            else:
                logger.warning(f"Key {key} missing from client {client_id}. Skipping.")

    logger.info("FedAvg aggregation complete.")
    return new_global_state