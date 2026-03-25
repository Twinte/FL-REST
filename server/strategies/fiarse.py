"""
FIARSE Server Strategy for FL-REST (CORRECTED)
=================================================
Importance-Aware Submodel Extraction via parameter magnitude.
(Wu et al., "FIARSE: Model-Heterogeneous FL via Importance-Aware
Submodel Extraction", NeurIPS 2024)

PAPER ALGORITHM (Algorithm 1):
  1. Server extracts submodel for client i by selecting parameters
     whose absolute value exceeds threshold θ_i = TopK_γi(|x̃|)
  2. Client trains using Threshold-Controlled Biased Gradient Descent
     (TCB-GD), which dynamically shrinks the mask during local training
  3. Client sends delta Δx = x̃_t - x^(i)_{t,K} back to server
  4. Server aggregates via partial averaging

KEY INSIGHT (from paper Section 4):
  Parameter magnitude IS the importance signal. No separate importance
  computation is needed. The TCB-GD biased gradient naturally pushes
  important parameters to grow and unimportant ones to shrink below
  threshold, making magnitudes increasingly reflective of true importance.

NOTE ON STRUCTURED VS UNSTRUCTURED:
  The paper operates at the individual parameter (weight) level —
  unstructured sparsity. Our framework uses neuron-level (structured)
  extraction. We adapt by computing per-neuron magnitude as the mean
  absolute weight, then selecting top-K neurons. Paper Section 7
  notes neuron-wise extraction as future work.
"""

import torch
import numpy as np
import logging
from .partial_training_base import PartialTrainingStrategy
from shared.submodel_utils import extract_submodel

logger = logging.getLogger(__name__)


class FIARSEStrategy(PartialTrainingStrategy):

    METHOD_NAME = "FIARSE"

    def __init__(self, global_model, ema_decay=0.9, total_rounds=100):
        """
        Args:
            global_model: PyTorch model instance.
            ema_decay: Accepted for factory compatibility but NOT USED.
                       Real FIARSE has no EMA — magnitudes are read fresh
                       from the global model each round.
            total_rounds: Total FL communication rounds.
        """
        super().__init__(global_model, total_rounds)
        # ema_decay intentionally unused — FIARSE uses live magnitudes
        logger.info(
            "FIARSE: Using magnitude-based top-K neuron extraction "
            "(adapted from per-parameter to per-neuron for structured pruning)"
        )

    # -----------------------------------------------------------------
    # Index computation: top-K neurons by global model magnitude
    # -----------------------------------------------------------------

    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        FIARSE extraction: select top floor(keep_ratio * N) neurons
        per layer, ranked by mean absolute weight magnitude of the
        current global model.

        This mirrors the paper's server-side extraction (Algorithm 1, Line 3):
          θ_i = TopK_γi(|x̃_t|)
          Send {x̃_t ⊙ M^(i)_t(x̃_t)} to client i

        Adapted from per-parameter to per-neuron granularity.
        """
        indices = {}
        for name, n_neurons in self.prunable_layers:
            param = dict(self.global_model.named_parameters())[name]

            # Per-neuron magnitude: mean |weight| across all connections
            if param.dim() == 4:  # Conv: [out_ch, in_ch, H, W]
                mag = param.data.abs().view(n_neurons, -1).mean(dim=1)
            elif param.dim() == 2:  # Linear: [out, in]
                mag = param.data.abs().mean(dim=1)
            else:
                mag = param.data.abs()

            n_keep = max(1, int(n_neurons * keep_ratio))
            top_idx = torch.argsort(mag, descending=True)[:n_keep]
            indices[name] = sorted(top_idx.cpu().tolist())

        return indices

    # -----------------------------------------------------------------
    # No get_payload_for_client override needed.
    #
    # The OLD (wrong) implementation overrode this to always send the
    # FULL model because clients needed it for gradient-based importance.
    # The CORRECTED FIARSE uses the base class behavior (compact submodel
    # when enabled, full model when not). FIARSE clients only need their
    # assigned neurons — TCB-GD operates within the submodel.
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # No special aggregation needed — base partial averaging is correct
    # (Paper Section 4.2, Line 13: standard partial averaging)
    # -----------------------------------------------------------------
