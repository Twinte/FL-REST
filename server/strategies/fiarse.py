"""
FIARSE Server Strategy for FL-REST
====================================
Client-side gradient importance with server-side EMA aggregation.
(Anonymous, NeurIPS 2024)

How it works:
  1. Each client computes grad x weight importance on local data
  2. Client uploads importance alongside model update (via tensor_metrics)
  3. Server aggregates all client importances via EMA
  4. Server uses aggregated importance for top-k neuron extraction

Key difference from FedPrune:
  - Importance source: CLIENT-SIDE (gradient-based) vs SERVER-SIDE (weight statistics)
  - Extra cost: Clients must run full-model forward+backward for importance
  - Privacy: Client importance scores leak information about local data

This is the strongest importance-aware baseline -- if FedPrune matches
FIARSE accuracy, it validates the server-side approach with zero client cost.
"""

import torch
import numpy as np
import logging
from .partial_training_base import PartialTrainingStrategy

logger = logging.getLogger(__name__)


class FIARSEStrategy(PartialTrainingStrategy):
    
    METHOD_NAME = "FIARSE"
    
    def __init__(self, global_model, ema_decay=0.9, total_rounds=100):
        super().__init__(global_model, total_rounds)
        
        self.ema_decay = ema_decay
        
        # EMA of aggregated client importance scores
        self.importance = {name: None for name, _ in self.prunable_layers}
        self.importance_initialized = False
        
        logger.info(f"FIARSE: Using client-side gradient importance, EMA decay={ema_decay}")
    
    # -----------------------------------------------------------------
    # Index computation: top-k by aggregated client importance
    # -----------------------------------------------------------------
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """Top-k neurons by aggregated client importance (or magnitude fallback)."""
        if not self.importance_initialized:
            return self._magnitude_fallback(keep_ratio)
        
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
        """Round 0 fallback before any client importance is received."""
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
    # Aggregation override: extract + aggregate client importance
    # -----------------------------------------------------------------
    
    def aggregate(self, updates):
        """
        Standard partial aggregation + extract and EMA-aggregate
        client-reported importance scores.
        """
        # Extract client importance from uploads before aggregation
        client_importances = []
        for update in updates:
            ci = update.get("metrics", {}).get("client_importance")
            if ci is not None:
                # Convert tensors to numpy if needed
                ci_np = {}
                for name, imp in ci.items():
                    if hasattr(imp, 'cpu'):
                        ci_np[name] = imp.cpu().numpy()
                    elif isinstance(imp, np.ndarray):
                        ci_np[name] = imp
                    else:
                        ci_np[name] = np.array(imp)
                client_importances.append(ci_np)
        
        # Run standard partial aggregation from base class
        result = super().aggregate(updates)
        
        # Aggregate client importances via EMA
        if client_importances:
            self._aggregate_importance(client_importances)
            logger.info(
                f"  FIARSE: Aggregated importance from {len(client_importances)} clients"
            )
        else:
            logger.warning("  FIARSE: No client importance received this round")
        
        return result
    
    def _aggregate_importance(self, client_importances):
        """EMA update of aggregated client importance."""
        for name, _ in self.prunable_layers:
            # Collect importances from clients that reported this layer
            layer_imps = [ci[name] for ci in client_importances if name in ci]
            if not layer_imps:
                continue
            
            fresh = np.mean(layer_imps, axis=0)
            
            if not self.importance_initialized or self.importance[name] is None:
                self.importance[name] = fresh
            else:
                self.importance[name] = (
                    self.ema_decay * self.importance[name] +
                    (1 - self.ema_decay) * fresh
                )
        
        self.importance_initialized = True

    def get_payload_for_client(self, client_id, model_state_dict):
        """
        FIARSE override: always send FULL model.
        
        FIARSE clients need the complete model for gradient-based importance
        computation (full forward+backward pass on all neurons). This is the
        method's fundamental cost — and what makes it communication-expensive
        compared to FedPrune.
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
        logger.info(
            f"  FIARSE: {client_id} payload {full_size/1024:.1f}KB "
            f"(FULL MODEL — required for importance computation)"
        )
        
        return {
            "model_state": model_state_dict,    # FULL — not extracted
            "extra_payload": clean_indices,
            "is_submodel": False,               # Client loads directly
        }