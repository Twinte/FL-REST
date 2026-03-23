"""
Partial Training Base Strategy for FL-REST
============================================
Shared infrastructure for all partial-training methods:
  - FedPrune (importance-based)
  - HeteroFL (first-k ordered)
  - FedRolex (rolling window)

Each subclass only needs to implement:
  _compute_indices_for_client(client_id, keep_ratio, round_num) -> dict

Everything else — capacity registration, partial aggregation,
per-client payload serving — is handled here.
"""

import torch
import numpy as np
import logging
from collections import OrderedDict
from .base import Strategy
from shared.submodel_utils import extract_submodel, reconstruct_full_state_from_upload

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Helpers
# =============================================================================

def get_prunable_layers(model):
    """
    Returns list of (name, n_neurons) for layers subject to pruning.
    Excludes: BatchNorm layers, bias terms, and the final classifier layer.
    """
    layers = []
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and param.dim() >= 2:
            if name == 'fc3.weight':
                continue
            layers.append((name, param.size(0)))
    return layers


def get_layer_sizes(model):
    """Returns {layer_name: n_neurons} for all weight layers (excl. BN)."""
    sizes = {}
    for name, param in model.named_parameters():
        if 'weight' in name and 'bn' not in name and param.dim() >= 2:
            sizes[name] = param.size(0)
    return sizes


# =============================================================================
# Base Class
# =============================================================================

class PartialTrainingStrategy(Strategy):
    """
    Abstract base for partial-training FL strategies.
    
    Lifecycle per round:
      1. compute_client_indices(client_ids) — called before model distribution
      2. get_payload_for_client(client_id, state_dict) — called per download
      3. aggregate(updates) — called when quorum is met
    
    Subclasses must implement:
      _compute_indices_for_client(client_id, keep_ratio, round_num) -> dict
    """
    
    METHOD_NAME = "PartialTraining"  # Override in subclasses
    
    def __init__(self, global_model, total_rounds=100):
        self.global_model = global_model
        self.total_rounds = total_rounds
        self.current_round = 0
        
        # Dynamic capacity dict — populated by register_capacity()
        self.client_capacities = {}
        
        # Layer topology (shared by all methods)
        self.prunable_layers = get_prunable_layers(global_model)
        self.layer_sizes = get_layer_sizes(global_model)
        
        # Per-round state
        self.round_indices = {}
        
        # Cumulative neuron coverage tracking (new metric)
        self.cumulative_coverage = {
            name: set() for name, _ in self.prunable_layers
        }
        
        logger.info(
            f"{self.METHOD_NAME} initialized: "
            f"{len(self.prunable_layers)} prunable layers"
        )
        for name, n in self.prunable_layers:
            logger.info(f"  Prunable: {name} ({n} neurons)")
    
    # -----------------------------------------------------------------
    # Capacity management
    # -----------------------------------------------------------------
    
    def register_capacity(self, client_id, capacity):
        """Register a client's capacity ratio. Called during /register."""
        clamped = min(1.0, max(0.1, capacity))
        self.client_capacities[client_id] = clamped
        logger.info(
            f"  Registered capacity: {client_id} \u2192 {clamped:.2f} "
            f"({len(self.client_capacities)} clients total)"
        )
    
    def _get_capacity(self, client_id):
        return self.client_capacities.get(client_id, 0.5)
    
    # -----------------------------------------------------------------
    # Index computation (template method — subclasses override the hook)
    # -----------------------------------------------------------------
    
    def compute_client_indices(self, participating_client_ids):
        """Compute neuron indices for all participating clients this round."""
        self.round_indices = {}
        
        for client_id in participating_client_ids:
            keep_ratio = self._get_capacity(client_id)
            indices = self._compute_indices_for_client(
                client_id, keep_ratio, self.current_round)
            
            # Always include full final classifier
            if 'fc3.weight' in self.layer_sizes and 'fc3.weight' not in indices:
                indices['fc3.weight'] = list(range(self.layer_sizes['fc3.weight']))
            
            self.round_indices[client_id] = indices
            logger.info(
                f"  {client_id}: ratio={keep_ratio:.2f}, "
                f"neurons={sum(len(v) for v in indices.values())}"
            )
        
        # --- NEW METRICS: Neuron overlap and coverage ---
        self._log_neuron_metrics(participating_client_ids)
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        OVERRIDE IN SUBCLASSES.
        
        Returns:
            dict: {layer_name: [neuron_indices]} for this client.
        """
        raise NotImplementedError
    
    def _log_neuron_metrics(self, participating_client_ids):
        """
        Log neuron overlap and cumulative coverage metrics.
        
        Metrics logged (parseable from logs):
          NEURON_OVERLAP: avg_jaccard=X.XXX across all client pairs
          NEURON_COVERAGE: X.X% of total neurons trained at least once
          TIER_OVERLAP: high_vs_low=X.XXX (Jaccard between capacity tiers)
        """
        if not self.round_indices:
            return
        
        # 1. Pairwise Jaccard overlap (first prunable layer as representative)
        first_layer = self.prunable_layers[0][0] if self.prunable_layers else None
        if first_layer:
            client_sets = {}
            for cid, indices in self.round_indices.items():
                if first_layer in indices:
                    client_sets[cid] = set(indices[first_layer])
            
            # Average pairwise Jaccard
            cids = list(client_sets.keys())
            jaccards = []
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    a, b = client_sets[cids[i]], client_sets[cids[j]]
                    if len(a | b) > 0:
                        jaccards.append(len(a & b) / len(a | b))
            
            avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
            logger.info(f"  NEURON_OVERLAP: avg_jaccard={avg_jaccard:.3f}")
            
            # Tier overlap: group by capacity, compare highest vs lowest tier
            tier_sets = {}
            for cid in cids:
                cap = self._get_capacity(cid)
                cap_key = f"{cap:.1f}"
                if cap_key not in tier_sets:
                    tier_sets[cap_key] = set()
                tier_sets[cap_key] |= client_sets[cid]
            
            caps_sorted = sorted(tier_sets.keys())
            if len(caps_sorted) >= 2:
                low_set = tier_sets[caps_sorted[0]]
                high_set = tier_sets[caps_sorted[-1]]
                if len(low_set | high_set) > 0:
                    tier_j = len(low_set & high_set) / len(low_set | high_set)
                    logger.info(
                        f"  TIER_OVERLAP: cap={caps_sorted[0]}_vs_{caps_sorted[-1]} "
                        f"jaccard={tier_j:.3f}"
                    )
        
        # 2. Cumulative coverage update
        for cid, indices in self.round_indices.items():
            for name, idx_list in indices.items():
                if name in self.cumulative_coverage:
                    self.cumulative_coverage[name].update(idx_list)
        
        total_neurons = sum(n for _, n in self.prunable_layers)
        covered = sum(len(s) for name, s in self.cumulative_coverage.items()
                      if any(name == pn for pn, _ in self.prunable_layers))
        coverage_pct = 100.0 * covered / total_neurons if total_neurons > 0 else 0.0
        logger.info(f"  NEURON_COVERAGE: {coverage_pct:.1f}% ({covered}/{total_neurons})")
    
    # -----------------------------------------------------------------
    # Per-client payload
    # -----------------------------------------------------------------
    
    def get_payload_for_client(self, client_id, model_state_dict):
        """
        Build a per-client payload with COMPACT submodel extraction.
        
        Instead of sending the full model + indices, we extract only the
        weights for assigned neurons. This reduces downlink by up to 15×
        for low-capacity clients.
        """
        indices = self.round_indices.get(client_id)
    
        if indices is None:
            logger.warning(
                f"No pre-computed indices for {client_id}. Computing on-the-fly."
            )
            keep_ratio = self._get_capacity(client_id)
            indices = self._compute_indices_for_client(
                client_id, keep_ratio, self.current_round)
            if 'fc3.weight' in self.layer_sizes and 'fc3.weight' not in indices:
                indices['fc3.weight'] = list(range(self.layer_sizes['fc3.weight']))
            self.round_indices[client_id] = indices
    
        clean_indices = {k: list(v) for k, v in indices.items()}
    
        # Extract compact submodel (the key communication optimization)
        #submodel_state = extract_submodel(model_state_dict, clean_indices)
    
        # Log the size reduction
        #full_size = sum(v.numel() * 4 for v in model_state_dict.values())
        #sub_size = sum(v.numel() * 4 for v in submodel_state.values())
        #logger.info(
        #    f"  {self.METHOD_NAME}: {client_id} payload "
        #   f"{sub_size/1024:.1f}KB / {full_size/1024:.1f}KB "
        #    f"({100*sub_size/full_size:.1f}%)"
        #)
        
        return {
            "model_state": model_state_dict, #submodel_state,
            "extra_payload": clean_indices,
            "is_submodel": False,
    }
    
    # -----------------------------------------------------------------
    # Partial aggregation (identical for all methods)
    # -----------------------------------------------------------------
    
    def aggregate(self, updates):
        """
        Partial aggregation: per-neuron weighted average from clients
        that trained each neuron. BatchNorm layers averaged across all.
        """
        if not updates:
            return None
        
        # Get reference state dict for shape information
        reference_state = self.global_model.state_dict()
    
        # Reconstruct compact uploads into full-sized state dicts
        for update in updates:
            client_id = update.get("client_id", "unknown")
            client_state = update["model_update"]
            
            if isinstance(client_state, dict) and client_state.get("is_submodel"):
                inner_state = client_state["model_state"]
                client_indices = self.round_indices.get(client_id, {})
                
                update["model_update"] = reconstruct_full_state_from_upload(
                    inner_state, client_indices, reference_state
                )
            elif isinstance(client_state, dict) and "model_state" in client_state:
                update["model_update"] = client_state["model_state"]
        
        client_states = []
        client_indices_list = []
        
        for update in updates:
            client_id = update.get("client_id", "unknown")
            client_states.append(update["model_update"])
            
            if client_id in self.round_indices:
                client_indices_list.append(self.round_indices[client_id])
            else:
                logger.warning(f"No index record for {client_id}; using full model.")
                all_indices = {n: list(range(s)) for n, s in self.layer_sizes.items()}
                client_indices_list.append(all_indices)
        
        global_state = self.global_model.state_dict()
        new_state = OrderedDict()
        
        for key in global_state.keys():
            param_shape = global_state[key].shape
            is_prunable = (
                'weight' in key and 'bn' not in key and
                key in client_indices_list[0]
            )
            
            if is_prunable:
                count = torch.zeros(param_shape[0], dtype=torch.float32)
                total = torch.zeros_like(global_state[key], dtype=torch.float32)
                
                for state, indices in zip(client_states, client_indices_list):
                    if key in indices:
                        idx = torch.tensor(indices[key], dtype=torch.long)
                        count[idx] += 1
                        total[idx] += state[key][idx].float()
                
                result = global_state[key].clone().float()
                mask = count > 0
                
                if mask.any():
                    if len(param_shape) == 4:
                        result[mask] = total[mask] / count[mask].view(-1, 1, 1, 1)
                    elif len(param_shape) == 2:
                        result[mask] = total[mask] / count[mask].view(-1, 1)
                    else:
                        result[mask] = total[mask] / count[mask]
                
                new_state[key] = result
            else:
                # Non-prunable (BN, bias, classifier): standard average
                stacked = torch.stack([s[key].float() for s in client_states])
                new_state[key] = stacked.mean(dim=0)
        
        self.global_model.load_state_dict(new_state)
        
        # Hook for subclasses (e.g., FedPrune updates EMA here)
        self._post_aggregation_hook(client_states)
        
        logger.info(
            f"{self.METHOD_NAME} aggregation complete (round {self.current_round}): "
            f"{len(updates)} clients"
        )
        
        self.current_round += 1
        
        return {
            "model_state": new_state,
            "extra_payload": None
        }
    
    def _post_aggregation_hook(self, client_states):
        """Override in subclasses that need post-aggregation processing."""
        pass

    # -----------------------------------------------------------------
    # Submodel Extraction (for communication reduction)
    # -----------------------------------------------------------------

    def _extract_submodel(self, full_state_dict, indices):
        """
        Extract only the weights corresponding to assigned neuron indices.
        Returns a compact state dict that is smaller on the wire.
        
        Args:
            full_state_dict: Complete model state dict
            indices: {layer_name: [neuron_indices]} for prunable layers
        
        Returns:
            dict with only the relevant slices of each tensor
        """
        submodel = {}
        
        layer_order = [name for name, _ in self.prunable_layers]
        prev_indices = None
        
        for layer_name, n_neurons in self.prunable_layers:
            if layer_name not in indices:
                continue
            
            out_idx = indices[layer_name]
            weight_key = layer_name
            bias_key = layer_name.replace('.weight', '.bias')
            
            w = full_state_dict[weight_key]
            
            w_sub = w[out_idx]
            
            if prev_indices is not None and w_sub.dim() >= 2:
                if w_sub.dim() == 4:
                    w_sub = w_sub[:, prev_indices]
                elif w_sub.dim() == 2:
                    if w_sub.shape[1] != len(prev_indices):
                        spatial_size = w_sub.shape[1] // self.layer_sizes.get(
                            layer_order[layer_order.index(layer_name) - 1], 
                            w_sub.shape[1]
                        )
                        if spatial_size > 1:
                            expanded = []
                            for idx in prev_indices:
                                expanded.extend(range(
                                    idx * spatial_size, 
                                    (idx + 1) * spatial_size
                                ))
                            w_sub = w_sub[:, expanded]
                        else:
                            w_sub = w_sub[:, prev_indices]
                    else:
                        w_sub = w_sub[:, prev_indices]
            
            submodel[weight_key] = w_sub
            
            if bias_key in full_state_dict:
                submodel[bias_key] = full_state_dict[bias_key][out_idx]
            
            bn_prefix = layer_name.replace('conv', 'bn').replace('.weight', '')
            for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
                bn_key = bn_prefix + suffix
                if bn_key in full_state_dict:
                    submodel[bn_key] = full_state_dict[bn_key][out_idx]
            bn_batches_key = bn_prefix + '.num_batches_tracked'
            if bn_batches_key in full_state_dict:
                submodel[bn_batches_key] = full_state_dict[bn_batches_key]
            
            prev_indices = out_idx
        
        fc3_w_key = 'fc3.weight'
        fc3_b_key = 'fc3.bias'
        if fc3_w_key in full_state_dict:
            fc3_w = full_state_dict[fc3_w_key]
            if prev_indices is not None:
                fc3_w = fc3_w[:, prev_indices]
            submodel[fc3_w_key] = fc3_w
        if fc3_b_key in full_state_dict:
            submodel[fc3_b_key] = full_state_dict[fc3_b_key]
        
        return submodel

    def _reconstruct_full_state(self, submodel_state, indices):
        """
        Reconstruct a full-sized state dict from a submodel upload.
        Non-trained neurons retain zeros (will be ignored during aggregation).
        """
        full_state = {}
        
        layer_order = [name for name, _ in self.prunable_layers]
        prev_indices = None
        
        for layer_name, n_neurons in self.prunable_layers:
            if layer_name not in indices:
                continue
            
            out_idx = indices[layer_name]
            weight_key = layer_name
            bias_key = layer_name.replace('.weight', '.bias')
            
            if weight_key in submodel_state:
                w_sub = submodel_state[weight_key]
                full_w = torch.zeros(n_neurons, w_sub.shape[1] if w_sub.dim() >= 2 else 1)
                
                if w_sub.dim() == 4:
                    full_w[:, prev_indices] = w_sub if prev_indices is not None else w_sub
                else:
                    full_w[out_idx] = w_sub
                
                full_state[weight_key] = full_w
            else:
                full_state[weight_key] = torch.zeros(n_neurons, self.layer_sizes.get(layer_name, 64))
            
            if bias_key in submodel_state:
                full_bias = torch.zeros(n_neurons)
                full_bias[out_idx] = submodel_state[bias_key]
                full_state[bias_key] = full_bias
            
            bn_prefix = layer_name.replace('conv', 'bn').replace('.weight', '')
            for suffix in ['.weight', '.bias', '.running_mean', '.running_var']:
                bn_key = bn_prefix + suffix
                if bn_key in submodel_state:
                    full_bn = torch.zeros(n_neurons)
                    full_bn[out_idx] = submodel_state[bn_key]
                    full_state[bn_key] = full_bn
            
            prev_indices = out_idx
        
        fc3_w_key = 'fc3.weight'
        fc3_b_key = 'fc3.bias'
        if fc3_w_key in submodel_state:
            full_state[fc3_w_key] = submodel_state[fc3_w_key]
        if fc3_b_key in submodel_state:
            full_state[fc3_b_key] = submodel_state[fc3_b_key]
        
        return full_state