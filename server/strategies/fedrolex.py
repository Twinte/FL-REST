"""
FedRolex Strategy for FL-REST
===============================
Rolling window neuron extraction (Alam et al., NeurIPS 2022).

Each round, the extraction window advances by n_keep positions,
cycling through all neurons. This ensures uniform coverage over
time but ignores neuron importance — every neuron gets equal
training frequency regardless of its contribution.
"""

import logging
from .partial_training_base import PartialTrainingStrategy

logger = logging.getLogger(__name__)


class FedRolexStrategy(PartialTrainingStrategy):
    
    METHOD_NAME = "FedRolex"
    
    def __init__(self, global_model, total_rounds=100):
        super().__init__(global_model, total_rounds)
        logger.info("FedRolex: Using rolling window extraction")
    
    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        Rolling window: start position advances by n_keep each round,
        wrapping around via modulo.
        """
        indices = {}
        for name, n_neurons in self.prunable_layers:
            n_keep = max(1, int(n_neurons * keep_ratio))
            start = (round_num * n_keep) % n_neurons
            idx = [(start + i) % n_neurons for i in range(n_keep)]
            indices[name] = sorted(idx)
        return indices