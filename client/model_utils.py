import torch
import io
import logging

logger = logging.getLogger(__name__)

def get_model_bytes(payload):
    """
    Serializes a payload (model dict or composite dict) to bytes.
    """
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    return buffer.read()

def set_model_from_bytes(model, model_bytes):
    """
    Loads bytes. If it's a SCAFFOLD payload, extracts the model first.
    Returns the 'extra_data' (e.g., global_c) if present.
    """
    buffer = io.BytesIO(model_bytes)
    try:
        # --- Enforce weights_only=True ---
        # This prevents Remote Code Execution (RCE) if the server is compromised.
        data = torch.load(buffer, map_location='cpu', weights_only=True)
        
        # Check if this is a composite payload (SCAFFOLD style)
        if isinstance(data, dict) and "model_state" in data:
            model.load_state_dict(data["model_state"])
            return data.get("extra_payload") # Return the Control Variate
        else:
            # It's just a standard model state_dict
            model.load_state_dict(data)
            return None
            
    except Exception as e:
        logger.error(f"Failed to load state_dict from bytes: {e}")
        raise