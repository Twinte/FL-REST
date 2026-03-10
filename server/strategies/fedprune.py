"""
FedPrune Strategy for FL-REST
==============================
Server-side importance estimation with hybrid importance-coverage extraction.

Extends PartialTrainingStrategy with:
  1. EMA tracking of cross-client weight variance and magnitude
  2. Neyman Allocation importance scoring (alpha * magnitude + (1-alpha) * variance)
  3. Capacity-adaptive hybrid extraction (importance exploit + rolling explore)
  4. Magnitude-based fallback for round 0 (before EMA is initialized)
"""

import torch
import numpy as np
import logging
from .partial_training_base import PartialTrainingStrategy, get_prunable_layers

logger = logging.getLogger(__name__)


# =============================================================================
# EMA Importance Tracker
# =============================================================================

class EMAImportanceTracker:
    """
    Tracks exponential moving averages of per-neuron magnitude and variance
    across FL rounds.
    """
    
    def __init__(self, prunable_layers, decay=0.9):
        self.decay = decay
        self.magnitude = {name: None for name, _ in prunable_layers}
        self.variance = {name: None for name, _ in prunable_layers}
        self.initialized = False
    
    def update(self, client_state_dicts, prunable_layers):
        K = len(client_state_dicts)
        if K == 0:
            return
        
        for name, n_neurons in prunable_layers:
            stacked = torch.stack([sd[name].float() for sd in client_state_dicts])
            flat = stacked.view(K, n_neurons, -1)
            
            fresh_mag = flat.abs().mean(dim=2).mean(dim=0).cpu().numpy()
            fresh_var = flat.var(dim=0).mean(dim=1).cpu().numpy()
            
            if not self.initialized:
                self.magnitude[name] = fresh_mag
                self.variance[name] = fresh_var
            else:
                self.magnitude[name] = (
                    self.decay * self.magnitude[name] +
                    (1 - self.decay) * fresh_mag
                )
                self.variance[name] = (
                    self.decay * self.variance[name] +
                    (1 - self.decay) * fresh_var
                )
        
        self.initialized = True
    
    def get_importance(self, alpha=0.3):
        eps = 1e-8
        importance = {}
        
        for name in self.magnitude:
            if self.magnitude[name] is None:
                continue
            mag = self.magnitude[name]
            var = self.variance[name]
            mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + eps)
            var_norm = (var - var.min()) / (var.max() - var.min() + eps)
            importance[name] = alpha * mag_norm + (1 - alpha) * var_norm
        
        return importance


# =============================================================================
# FedPrune Strategy
# =============================================================================

class FedPruneStrategy(PartialTrainingStrategy):
    
    METHOD_NAME = "FedPrune"
    
    def __init__(self, global_model, ema_decay=0.9, importance_alpha=0.3,
                 total_rounds=100, ramp_range=0.0):
        super().__init__(global_model, total_rounds)
        
        self.importance_alpha = importance_alpha
        self.ramp_range = ramp_range
        self.ema_tracker = EMAImportanceTracker(self.prunable_layers, decay=ema_decay)
        
        schedule = "round-adaptive" if ramp_range > 0 else "static"
        logger.info(
            f"FedPrune: alpha={importance_alpha}, decay={ema_decay}, "
            f"ramp={ramp_range} ({schedule})"
        )
    
    # -----------------------------------------------------------------
    # Override: index computation hook
    # -----------------------------------------------------------------
    
    def compute_client_indices(self, participating_client_ids):
        """Override to log capacity-adaptive imp_frac before computing."""
        imp_frac = self._compute_imp_frac(participating_client_ids)
        # Store for use in per-client computation
        self._current_imp_frac = imp_frac
        # Delegate to base class (which calls _compute_indices_for_client per client)
        super().compute_client_indices(participating_client_ids)
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        FedPrune extraction:
          - Round 0 (EMA uninitialized): magnitude-based fallback
          - Round 1+: hybrid importance + rolling coverage
        """
        if not self.ema_tracker.initialized:
            return self._magnitude_fallback(keep_ratio)
        
        importance = self.ema_tracker.get_importance(alpha=self.importance_alpha)
        imp_frac = getattr(self, '_current_imp_frac', 0.5)
        return self._hybrid_extraction(importance, keep_ratio, round_num, imp_frac)
    
    # -----------------------------------------------------------------
    # Extraction methods
    # -----------------------------------------------------------------
    
    def _magnitude_fallback(self, keep_ratio):
        """Round 0 fallback: select neurons by weight magnitude."""
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
    
    def _hybrid_extraction(self, importance, keep_ratio, round_num, imp_frac):
        """
        Hybrid importance-coverage extraction.
        Splits neuron budget: imp_frac by importance, remainder by rolling window.
        """
        indices = {}
        for name, imp in importance.items():
            n_total = len(imp)
            n_keep = max(1, int(n_total * keep_ratio))
            n_imp = max(1, int(n_keep * imp_frac))
            n_roll = n_keep - n_imp
            
            imp_idx = set(np.argsort(imp)[-n_imp:].tolist())
            
            roll_idx = set()
            if n_roll > 0:
                start = (round_num * n_roll) % n_total
                for i in range(n_total):
                    candidate = (start + i) % n_total
                    if candidate not in imp_idx:
                        roll_idx.add(candidate)
                        if len(roll_idx) >= n_roll:
                            break
            
            indices[name] = sorted(list(imp_idx | roll_idx))
        return indices
    
    # -----------------------------------------------------------------
    # Capacity-adaptive imp_frac
    # -----------------------------------------------------------------
    
    def _compute_imp_frac(self, participating_client_ids):
        """
        Round-adaptive imp_frac (Phase 5v3).
        
        Base:     clamp(0.5 + std(capacities), 0.5, 0.9)   [capacity-adaptive]
        Schedule: base - ramp/2 + progress * ramp            [round-adaptive]
        
        With default ramp=0.3 and cap_std=0.15 (base=0.65):
          Round 0:   imp_frac = 0.50  (explore: broad coverage, EMA unreliable)
          Round 50:  imp_frac = 0.65  (balanced)
          Round 99:  imp_frac = 0.80  (exploit: importance-heavy, EMA stable)
        """
        capacities = [self._get_capacity(c) for c in participating_client_ids]
        cap_std = float(np.std(capacities))
        base_frac = min(0.9, max(0.5, 0.5 + cap_std))
        
        # Round-adaptive ramp: explore early, exploit late
        progress = self.current_round / max(self.total_rounds - 1, 1)
        imp_frac = base_frac - self.ramp_range / 2 + progress * self.ramp_range
        imp_frac = min(0.9, max(0.3, imp_frac))
        
        logger.info(
            f"  Capacity std={cap_std:.3f}, base={base_frac:.2f}, "
            f"progress={progress:.2f} \u2192 imp_frac={imp_frac:.2f}"
        )
        return imp_frac
    
    # -----------------------------------------------------------------
    # Post-aggregation: update EMA tracker
    # -----------------------------------------------------------------
    
    def _post_aggregation_hook(self, client_states):
        """Update EMA importance tracker with this round's client weights."""
        self.ema_tracker.update(client_states, self.prunable_layers)
        logger.info(
            f"  EMA initialized={self.ema_tracker.initialized}"
        )