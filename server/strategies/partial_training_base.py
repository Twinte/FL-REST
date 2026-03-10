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
        """Build a per-client composite payload (model + indices)."""
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
        
        return {
            "model_state": model_state_dict,
            "extra_payload": clean_indices
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
                        for idx in indices[key]:
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