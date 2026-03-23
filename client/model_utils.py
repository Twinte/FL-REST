"""
Client Model Utilities for FL-REST
=====================================
Handles serialization/deserialization of model payloads, including
compact submodel reconstruction.

REPLACES: client/model_utils.py
"""

import torch
import io
import logging

logger = logging.getLogger(__name__)

# Import shared submodel logic
from shared.submodel_utils import load_submodel_into_model, extract_trained_submodel


def get_model_bytes(payload):
    """
    Serializes a payload (model dict or composite dict) to bytes.
    Unchanged from original.
    """
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return buffer.read()


def set_model_from_bytes(model, model_bytes):
    """
    Loads model bytes into the model. Handles three payload types:
    
    1. Compact submodel (is_submodel=True):
       - Places submodel weights into correct positions of existing model
       - Non-assigned neurons retain their values from previous round
       - Returns neuron indices for gradient masking
    
    2. Full composite payload (is_submodel=False, has extra_payload):
       - Loads full state dict directly (used by FIARSE, FLuID leaders)
       - Returns extra_payload (neuron indices or SCAFFOLD control variate)
    
    3. Plain state dict (legacy):
       - Loads directly, returns None
    """
    buffer = io.BytesIO(model_bytes)
    try:
        data = torch.load(buffer, map_location='cpu', weights_only=True)
        
        if isinstance(data, dict) and "model_state" in data:
            is_submodel = data.get("is_submodel", False)
            extra_payload = data.get("extra_payload")
            
            if is_submodel and extra_payload:
                # Compact submodel — reconstruct into full model
                load_submodel_into_model(model, data["model_state"], extra_payload)
                
                sub_params = sum(v.numel() for v in data["model_state"].values())
                full_params = sum(p.numel() for p in model.parameters())
                logger.info(
                    f"Loaded submodel: {sub_params:,} params "
                    f"({100*sub_params/full_params:.1f}% of full)"
                )
            else:
                # Full model — load directly
                model.load_state_dict(data["model_state"])
            
            return extra_payload
        else:
            # Plain state dict (legacy path)
            model.load_state_dict(data)
            return None
            
    except Exception as e:
        logger.error(f"Failed to load state_dict from bytes: {e}")
        raise


def prepare_upload_payload(model, neuron_indices=None, method_name=""):
    """
    Prepare the model update for upload.
    
    For partial-training methods (except FIARSE which also uploads importance):
      - Extract only trained neurons for compact upload
    
    For full-model methods or FIARSE:
      - Upload the full state dict
    
    Args:
        model: Trained PyTorch model
        neuron_indices: Dict {layer_name: [indices]} or None
        method_name: Algorithm name (to decide upload strategy)
    
    Returns:
        bytes: Serialized payload ready for HTTP upload
    """
    if neuron_indices and method_name not in ("FIARSE",):
        # Compact upload: extract only trained neurons
        submodel_state = extract_trained_submodel(model, neuron_indices)
        
        payload = {
            "model_state": submodel_state,
            "is_submodel": True,
        }
        
        model_bytes = get_model_bytes(payload)
        
        full_size = sum(p.numel() * 4 for p in model.parameters())
        logger.info(
            f"Upload: compact submodel {len(model_bytes)} bytes "
            f"({100*len(model_bytes)/full_size:.1f}% of full model)"
        )
    else:
        # Full upload (standard or FIARSE)
        model_bytes = get_model_bytes(model.state_dict())
    
    return model_bytes