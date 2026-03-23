"""
FLuID Server Strategy for FL-REST
====================================
Leader-client invariant dropout (Yang et al., AAAI 2023).

How it works:
  1. Server designates high-capacity clients as "leaders"
  2. Leaders receive the FULL model (all neurons, no pruning)
  3. Stragglers receive pruned submodels based on importance
  4. After leaders submit, server computes importance from
     their update deltas: |leader_update - global_model|
  5. Uses delta-based importance for straggler extraction next round

Key assumptions:
  - Requires a subset of "leader" clients to always be available
  - Leaders must have enough capacity to train the full model
  - Importance is biased toward leader data distributions
  - Fails gracefully if no leaders participate (falls back to magnitude)

Key difference from FedPrune:
  - Importance source: LEADER CLIENTS (update deltas)
  - Dependency: Requires leader availability each round
  - Bias: Importance reflects leader data, not population consensus
"""

import torch
import numpy as np
import logging
from .partial_training_base import PartialTrainingStrategy
from shared.submodel_utils import extract_submodel

logger = logging.getLogger(__name__)


# Clients with capacity >= this threshold become leaders
LEADER_CAPACITY_THRESHOLD = 0.6


class FLuIDStrategy(PartialTrainingStrategy):
    
    METHOD_NAME = "FLuID"
    
    def __init__(self, global_model, ema_decay=0.9,
                 leader_threshold=LEADER_CAPACITY_THRESHOLD,
                 total_rounds=100):
        super().__init__(global_model, total_rounds)
        
        self.ema_decay = ema_decay
        self.leader_threshold = leader_threshold
        
        # EMA of leader-derived importance
        self.importance = {name: None for name, _ in self.prunable_layers}
        self.importance_initialized = False
        
        # Track which clients are leaders
        self.leader_clients = set()
        
        logger.info(
            f"FLuID: leader_threshold={leader_threshold}, "
            f"EMA decay={ema_decay}"
        )
    
    # -----------------------------------------------------------------
    # Capacity registration: auto-detect leaders
    # -----------------------------------------------------------------
    
    def register_capacity(self, client_id, capacity):
        """Register capacity and classify as leader or straggler."""
        super().register_capacity(client_id, capacity)
        
        if capacity >= self.leader_threshold:
            self.leader_clients.add(client_id)
            logger.info(f"  FLuID: {client_id} designated as LEADER (cap={capacity:.2f})")
        else:
            self.leader_clients.discard(client_id)
            logger.info(f"  FLuID: {client_id} designated as STRAGGLER (cap={capacity:.2f})")
    
    # -----------------------------------------------------------------
    # Index computation: leaders get full, stragglers get importance-based
    # -----------------------------------------------------------------
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        Leaders: all neurons (full model training).
        Stragglers: top-k by leader-derived importance (or magnitude fallback).
        """
        if client_id in self.leader_clients:
            return self._full_model_indices()
        
        # Straggler path
        if not self.importance_initialized:
            return self._magnitude_fallback(keep_ratio)
        
        return self._importance_extraction(keep_ratio)
    
    def _full_model_indices(self):
        """All neurons for leader clients."""
        indices = {}
        for name, n_neurons in self.prunable_layers:
            indices[name] = list(range(n_neurons))
        return indices
    
    def _importance_extraction(self, keep_ratio):
        """Top-k by leader-derived importance."""
        indices = {}
        for name, _ in self.prunable_layers:
            imp = self.importance[name]
            if imp is None:
                continue
            n_total = len(imp)
            n_keep = max(1, int(n_total * keep_ratio))
            top_idx = np.argsort(imp)[-n_keep:]
            indices[name] = sorted(top_idx.tolist())
        return indices
    
    def _magnitude_fallback(self, keep_ratio):
        """Pre-importance fallback: select by weight magnitude."""
        indices = {}
        for name, n_neurons in self.prunable_layers:
            param = dict(self.global_model.named_parameters())[name]
            if param.dim() == 4:
                mag = param.data.view(n_neurons, -1).abs().mean(dim=1)
            else:
                mag = param.data.abs().mean(dim=1)
            n_keep = max(1, int(n_neurons * keep_ratio))
            top_idx = torch.argsort(mag, descending=True)[:n_keep].cpu().tolist()
            indices[name] = sorted(top_idx)
        return indices
    
    # -----------------------------------------------------------------
    # Aggregation override: capture pre-agg state for delta computation
    # -----------------------------------------------------------------
    
    def aggregate(self, updates):
        """Capture pre-aggregation global state, then delegate to base."""
        # Save pre-aggregation weights for leader delta computation
        self._pre_agg_state = {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
            if 'weight' in name and 'bn' not in name and param.dim() >= 2
        }
        
        # Track which client_ids participated (in order) for leader matching
        self._participating_ids = [u.get("client_id", "unknown") for u in updates]
        
        return super().aggregate(updates)
    
    # -----------------------------------------------------------------
    # Post-aggregation: compute importance from leader deltas
    # -----------------------------------------------------------------
    
    def _post_aggregation_hook(self, client_states):
        """
        Compute importance from leader update deltas.
        
        delta = |leader_weights - pre_aggregation_global_weights|
        Importance = EMA of mean(leader_deltas) per neuron.
        """
        pre_agg = getattr(self, '_pre_agg_state', None)
        participating_ids = getattr(self, '_participating_ids', [])
        
        if pre_agg is None:
            return
        
        try:
            # Identify leader states
            leader_states = []
            for i, state in enumerate(client_states):
                if i < len(participating_ids):
                    cid = participating_ids[i]
                    if cid in self.leader_clients:
                        leader_states.append(state)
            
            if not leader_states:
                logger.info("  FLuID: No leaders participated this round, skipping importance update")
                return
            
            # Compute delta-based importance from leader updates
            for name, n_neurons in self.prunable_layers:
                if name not in pre_agg:
                    continue
                
                deltas = []
                for ls in leader_states:
                    delta = (ls[name].float() - pre_agg[name].float()).abs()
                    if delta.dim() == 4:
                        deltas.append(delta.view(n_neurons, -1).mean(dim=1).cpu().numpy())
                    elif delta.dim() == 2:
                        deltas.append(delta.mean(dim=1).cpu().numpy())
                    else:
                        deltas.append(delta.cpu().numpy())
                
                fresh = np.mean(deltas, axis=0)
                
                if not self.importance_initialized or self.importance[name] is None:
                    self.importance[name] = fresh
                else:
                    self.importance[name] = (
                        self.ema_decay * self.importance[name] +
                        (1 - self.ema_decay) * fresh
                    )
            
            self.importance_initialized = True
            logger.info(
                f"  FLuID: Updated importance from {len(leader_states)} leaders"
            )
        except Exception as e:
            logger.error(f"  FLuID: Error in post-aggregation hook: {e}", exc_info=True)
        finally:
            # Free pre-aggregation state to avoid memory accumulation
            self._pre_agg_state = None
            self._participating_ids = None

    def get_payload_for_client(self, client_id, model_state_dict):
        """
        FLuID override: leaders get full model, stragglers get submodel.
        
        Leaders must train the full model so their update deltas can be
        used to compute importance for stragglers. This is FLuID's
        structural dependency — and its vulnerability if leaders drop.
        """
        indices = self.round_indices.get(client_id)
        
        if indices is None:
            keep_ratio = self._get_capacity(client_id)
            indices = self._compute_indices_for_client(
                client_id, keep_ratio, self.current_round)
            if 'fc3.weight' in self.layer_sizes and 'fc3.weight' not in indices:
                indices['fc3.weight'] = list(range(self.layer_sizes['fc3.weight']))
            self.round_indices[client_id] = indices
        
        clean_indices = {k: list(v) for k, v in indices.items()}
        
        full_size = sum(v.numel() * 4 for v in model_state_dict.values())
        
        if client_id in self.leader_clients:
            # Leaders: full model (they train everything)
            logger.info(
                f"  FLuID: {client_id} (LEADER) payload {full_size/1024:.1f}KB "
                f"(FULL MODEL)"
            )
            return {
                "model_state": model_state_dict,
                "extra_payload": clean_indices,
                "is_submodel": False,
            }
        else:
            # Stragglers: compact submodel
            submodel_state = extract_submodel(model_state_dict, clean_indices)
            sub_size = sum(v.numel() * 4 for v in submodel_state.values())
            logger.info(
                f"  FLuID: {client_id} (STRAGGLER) payload "
                f"{sub_size/1024:.1f}KB / {full_size/1024:.1f}KB "
                f"({100*sub_size/full_size:.1f}%)"
            )
            return {
                "model_state": submodel_state,
                "extra_payload": clean_indices,
                "is_submodel": True,
            }