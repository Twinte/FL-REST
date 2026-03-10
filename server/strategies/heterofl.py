"""
HeteroFL Strategy for FL-REST
==============================
First-k ordered neuron extraction (Diao et al., ICLR 2021).

Always selects the first floor(capacity * N) neurons per layer.
No importance signal — neuron selection is static and arbitrary.

This creates a structural bias: early neurons are always trained,
later neurons are undertrained proportional to the capacity mix.
"""

import logging
from .partial_training_base import PartialTrainingStrategy

logger = logging.getLogger(__name__)


class HeteroFLStrategy(PartialTrainingStrategy):
    
    METHOD_NAME = "HeteroFL"
    
    def __init__(self, global_model, total_rounds=100):
        super().__init__(global_model, total_rounds)
        logger.info("HeteroFL: Using first-k ordered extraction")
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """Select first floor(keep_ratio * N) neurons per prunable layer."""
        indices = {}
        for name, n_neurons in self.prunable_layers:
            n_keep = max(1, int(n_neurons * keep_ratio))
            indices[name] = list(range(n_keep))
        return indices