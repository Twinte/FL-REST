"""
Submodel Extraction & Reconstruction for FL-REST
===================================================
Converts between full model state_dicts and compact submodel state_dicts
based on neuron index assignments.

Used by:
  - Server: extract submodel before sending to client (smaller downlink)
  - Client: reconstruct full model from submodel (place weights correctly)
  - Client: extract trained submodel before uploading (smaller uplink)

The key complexity is tracking input dimensions across the layer chain:
  conv1 (in=3 fixed) → conv2 (in=conv1_out) → conv3 (in=conv2_out)
  → fc1 (in=conv3_out * spatial_size) → fc2 (in=fc1_out) → fc3 (in=fc2_out, out=10 fixed)

Supports the FedPruneCNN / SimpleCNN architecture:
  conv1[32] → bn1 → conv2[64] → bn2 → conv3[128] → bn3
  → fc1[256] → fc2[128] → fc3[10]
"""

import torch
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Architecture Definition
# =============================================================================
# Ordered list of prunable layers and their relationships.
# This must match get_prunable_layers() in partial_training_base.py.
#
# Format: (weight_key, bn_prefix_or_None, is_conv_to_fc_transition)
#
# The conv→fc transition requires special handling because the flattening
# operation expands each conv filter into spatial_size input positions.

LAYER_CHAIN = [
    # (weight_key,    bn_prefix,  conv_to_fc_boundary)
    ("conv1.weight",  "bn1",      False),
    ("conv2.weight",  "bn2",      False),
    ("conv3.weight",  "bn3",      False),
    ("fc1.weight",    None,       True),    # input from conv3 flattened
    ("fc2.weight",    None,       False),
]

CLASSIFIER_KEY = "fc3.weight"
CLASSIFIER_BIAS = "fc3.bias"


def _get_spatial_size(full_state_dict, conv_key, fc_key):
    """
    Compute the spatial flattening factor at the conv→fc boundary.
    
    If conv3 has N filters and fc1 has input dim D, then:
      spatial_size = D / N
    
    For FedPruneCNN with conv3[128] and fc1 input 2048:
      spatial_size = 2048 / 128 = 16  (i.e., 4×4 feature map)
    """
    conv_out = full_state_dict[conv_key].shape[0]
    fc_in = full_state_dict[fc_key].shape[1]
    spatial = fc_in // conv_out
    return spatial


def _expand_conv_indices_to_fc(conv_indices, spatial_size):
    """
    Expand conv filter indices into flattened fc input indices.
    
    Each conv filter index i maps to positions:
      [i * spatial_size, i * spatial_size + 1, ..., i * spatial_size + spatial_size - 1]
    
    Example: conv indices [2, 5] with spatial_size=16 →
      [32,33,...,47, 80,81,...,95]
    """
    expanded = []
    for idx in conv_indices:
        start = idx * spatial_size
        expanded.extend(range(start, start + spatial_size))
    return expanded


# =============================================================================
# Extraction: Full State Dict → Compact Submodel
# =============================================================================

def extract_submodel(full_state_dict, indices):
    """
    Extract a compact submodel containing only the weights for assigned neurons.
    
    Args:
        full_state_dict: Complete model state_dict (all layers, full dimensions)
        indices: Dict {prunable_layer_name: [neuron_indices]}
                 e.g., {"conv1.weight": [0,3,7], "conv2.weight": [1,5,10,15], ...}
    
    Returns:
        dict: Compact state dict with reduced tensor dimensions.
              Much smaller than full_state_dict when capacity < 1.0.
    """
    submodel = {}
    prev_out_indices = None   # Track output indices for input dim of next layer
    prev_layer_key = None     # Track which layer produced prev_out_indices
    
    for weight_key, bn_prefix, is_conv_to_fc in LAYER_CHAIN:
        if weight_key not in indices:
            # Layer not in assignment — skip (shouldn't happen in normal use)
            logger.warning(f"extract_submodel: {weight_key} not in indices, skipping")
            continue
        
        out_idx = indices[weight_key]
        bias_key = weight_key.replace(".weight", ".bias")
        
        w = full_state_dict[weight_key]
        
        # ----- Output dimension slicing -----
        w_sub = w[out_idx]
        
        # ----- Input dimension slicing -----
        if prev_out_indices is not None and w_sub.dim() >= 2:
            if is_conv_to_fc:
                # Conv→FC transition: expand conv indices to flattened positions
                spatial = _get_spatial_size(full_state_dict, prev_layer_key, weight_key)
                in_idx = _expand_conv_indices_to_fc(prev_out_indices, spatial)
                w_sub = w_sub[:, in_idx]
            else:
                # Conv→Conv or FC→FC: direct input indexing
                w_sub = w_sub[:, prev_out_indices]
        
        submodel[weight_key] = w_sub
        
        # ----- Bias (indexed by output only) -----
        if bias_key in full_state_dict:
            submodel[bias_key] = full_state_dict[bias_key][out_idx]
        
        # ----- BatchNorm parameters -----
        if bn_prefix is not None:
            for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
                bn_key = bn_prefix + suffix
                if bn_key in full_state_dict:
                    submodel[bn_key] = full_state_dict[bn_key][out_idx]
            
            # num_batches_tracked is a scalar — always include
            nbt_key = bn_prefix + ".num_batches_tracked"
            if nbt_key in full_state_dict:
                submodel[nbt_key] = full_state_dict[nbt_key]
        
        # ----- Advance chain -----
        prev_out_indices = out_idx
        prev_layer_key = weight_key
    
    # ----- Final classifier (fc3): full output, reduced input -----
    if CLASSIFIER_KEY in full_state_dict:
        fc3_w = full_state_dict[CLASSIFIER_KEY]
        if prev_out_indices is not None:
            fc3_w = fc3_w[:, prev_out_indices]
        submodel[CLASSIFIER_KEY] = fc3_w
    
    if CLASSIFIER_BIAS in full_state_dict:
        submodel[CLASSIFIER_BIAS] = full_state_dict[CLASSIFIER_BIAS]
    
    return submodel


# =============================================================================
# Reconstruction: Compact Submodel → Placed Into Full Model
# =============================================================================

def load_submodel_into_model(model, submodel_state, indices):
    """
    Place compact submodel weights into the correct positions of an existing
    model. Non-assigned neurons retain their values from the previous round.
    
    Args:
        model: PyTorch model (full-sized, with existing weights)
        submodel_state: Compact state dict from extract_submodel()
        indices: Dict {prunable_layer_name: [neuron_indices]}
    """
    current_state = model.state_dict()
    
    device = next(iter(current_state.values())).device
    submodel_state = {k: v.to(device) if hasattr(v, 'to') else v 
                      for k, v in submodel_state.items()}
    prev_out_indices = None
    prev_layer_key = None
    
    for weight_key, bn_prefix, is_conv_to_fc in LAYER_CHAIN:
        if weight_key not in indices:
            continue
        
        out_idx = indices[weight_key]
        bias_key = weight_key.replace(".weight", ".bias")
        
        # ----- Reconstruct weight tensor -----
        if weight_key in submodel_state:
            full_w = current_state[weight_key].clone()
            sub_w = submodel_state[weight_key]
            
            if prev_out_indices is not None:
                if is_conv_to_fc:
                    spatial = _get_spatial_size(current_state, prev_layer_key, weight_key)
                    in_idx = _expand_conv_indices_to_fc(prev_out_indices, spatial)
                else:
                    in_idx = prev_out_indices
                
                out_t = torch.tensor(out_idx, dtype=torch.long)
                in_t = torch.tensor(in_idx, dtype=torch.long)
                
                if sub_w.dim() == 4:
                    full_w[out_t[:, None], in_t[None, :], :, :] = sub_w
                elif sub_w.dim() == 2:
                    full_w[out_t[:, None], in_t[None, :]] = sub_w
                else:
                    full_w[out_t] = sub_w
            else:
                full_w[out_idx] = sub_w
            
            current_state[weight_key] = full_w
        
        # ----- Reconstruct bias -----
        if bias_key in submodel_state and bias_key in current_state:
            full_b = current_state[bias_key].clone()
            full_b[out_idx] = submodel_state[bias_key]
            current_state[bias_key] = full_b
        
        # ----- Reconstruct BatchNorm -----
        if bn_prefix is not None:
            for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
                bn_key = bn_prefix + suffix
                if bn_key in submodel_state and bn_key in current_state:
                    full_bn = current_state[bn_key].clone()
                    full_bn[out_idx] = submodel_state[bn_key]
                    current_state[bn_key] = full_bn
            
            nbt_key = bn_prefix + ".num_batches_tracked"
            if nbt_key in submodel_state:
                current_state[nbt_key] = submodel_state[nbt_key]
        
        prev_out_indices = out_idx
        prev_layer_key = weight_key
    
    # ----- Final classifier -----
    if CLASSIFIER_KEY in submodel_state and CLASSIFIER_KEY in current_state:
        full_fc3 = current_state[CLASSIFIER_KEY].clone()
        sub_fc3 = submodel_state[CLASSIFIER_KEY]
        if prev_out_indices is not None:
            full_fc3[:, prev_out_indices] = sub_fc3
        else:
            full_fc3 = sub_fc3
        current_state[CLASSIFIER_KEY] = full_fc3
    
    if CLASSIFIER_BIAS in submodel_state:
        current_state[CLASSIFIER_BIAS] = submodel_state[CLASSIFIER_BIAS]
    
    model.load_state_dict(current_state)


# =============================================================================
# Client-Side Upload Extraction
# =============================================================================

def extract_trained_submodel(model, indices):
    """
    Extract only the output neurons this client trained, for compact upload.
    
    IMPORTANT: We only slice the OUTPUT dimension, not the input dimension.
    Gradient masking preserves full input dimensions in the trained model,
    so the upload must include the complete rows for correct aggregation.
    """
    state = model.state_dict()
    submodel = {}
    
    for weight_key, bn_prefix, is_conv_to_fc in LAYER_CHAIN:
        if weight_key not in indices:
            continue
        
        out_idx = indices[weight_key]
        bias_key = weight_key.replace(".weight", ".bias")
        
        # Only slice output dimension — keep full input dimension
        submodel[weight_key] = state[weight_key][out_idx]
        
        if bias_key in state:
            submodel[bias_key] = state[bias_key][out_idx]
        
        if bn_prefix is not None:
            for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
                bn_key = bn_prefix + suffix
                if bn_key in state:
                    submodel[bn_key] = state[bn_key][out_idx]
            nbt_key = bn_prefix + ".num_batches_tracked"
            if nbt_key in state:
                submodel[nbt_key] = state[nbt_key]
    
    # Final classifier: full (always trained entirely)
    if CLASSIFIER_KEY in state:
        submodel[CLASSIFIER_KEY] = state[CLASSIFIER_KEY]
    if CLASSIFIER_BIAS in state:
        submodel[CLASSIFIER_BIAS] = state[CLASSIFIER_BIAS]
    
    return submodel


# =============================================================================
# Server-Side Upload Reconstruction
# =============================================================================

def reconstruct_full_state_from_upload(submodel_upload, indices, reference_state_dict):
    """
    Reconstruct full state dict from compact upload.
    Upload only has output-sliced rows with full input dimensions.
    """
    full_state = {}
    for key, ref_tensor in reference_state_dict.items():
        full_state[key] = ref_tensor.clone()
    
    device = next(iter(reference_state_dict.values())).device
    submodel_upload = {k: v.to(device) if hasattr(v, 'to') else v 
                       for k, v in submodel_upload.items()}
    
    for weight_key, bn_prefix, is_conv_to_fc in LAYER_CHAIN:
        if weight_key not in indices:
            continue
        
        out_idx = indices[weight_key]
        bias_key = weight_key.replace(".weight", ".bias")
        
        if weight_key in submodel_upload:
            full_state[weight_key][out_idx] = submodel_upload[weight_key]
        
        if bias_key in submodel_upload and bias_key in full_state:
            full_state[bias_key][out_idx] = submodel_upload[bias_key]
        
        if bn_prefix is not None:
            for suffix in (".weight", ".bias", ".running_mean", ".running_var"):
                bn_key = bn_prefix + suffix
                if bn_key in submodel_upload and bn_key in full_state:
                    full_state[bn_key][out_idx] = submodel_upload[bn_key]
            nbt_key = bn_prefix + ".num_batches_tracked"
            if nbt_key in submodel_upload:
                full_state[nbt_key] = submodel_upload[nbt_key]
    
    if CLASSIFIER_KEY in submodel_upload:
        full_state[CLASSIFIER_KEY] = submodel_upload[CLASSIFIER_KEY]
    if CLASSIFIER_BIAS in submodel_upload:
        full_state[CLASSIFIER_BIAS] = submodel_upload[CLASSIFIER_BIAS]
    
    return full_state


# =============================================================================
# Utilities
# =============================================================================

def compute_submodel_size(submodel_state):
    """Compute the parameter count and byte size of a submodel state dict."""
    total_params = 0
    for key, tensor in submodel_state.items():
        total_params += tensor.numel()
    total_bytes = total_params * 4  # FP32
    return total_params, total_bytes