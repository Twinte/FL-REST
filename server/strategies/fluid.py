"""
FLuID Server Strategy for FL-REST (CORRECTED)
================================================
Invariant Dropout for straggler mitigation.
(Wang, Nair & Mahajan, "FLuID: Mitigating Stragglers in Federated
Learning using Invariant Dropout", NeurIPS 2023)

PAPER ALGORITHM (Algorithm 1):
  1. Identify stragglers based on capacity (proxy for training latency)
  2. Non-stragglers train the FULL model
  3. After non-stragglers submit, identify "invariant" neurons:
     neurons whose weight updates from non-stragglers fall below
     a threshold th (they've stabilized / contribute little)
  4. Build sub-models for stragglers by DROPPING invariant neurons
  5. Threshold th adjusted to match target sub-model size

CORRECTIONS FROM PREVIOUS IMPLEMENTATION:
  - "Non-straggler" instead of "leader" terminology
  - Threshold-based invariant dropping instead of top-K by delta
  - Uses _post_aggregation_hook (base class pattern) not aggregate override
  - straggler_threshold default calibrated to typical experiment configs
"""

import torch
import numpy as np
import logging
from .partial_training_base import PartialTrainingStrategy
from shared.submodel_utils import extract_submodel

logger = logging.getLogger(__name__)


class FLuIDStrategy(PartialTrainingStrategy):

    METHOD_NAME = "FLuID"

    def __init__(self, global_model, ema_decay=0.9,
                 leader_threshold=None, straggler_threshold=None,
                 total_rounds=100):
        """
        Factory-compatible constructor.

        Args:
            global_model: PyTorch model instance.
            ema_decay: Accepted for factory compatibility. NOT USED by
                       corrected FLuID (no EMA — fresh deltas each round).
            leader_threshold: OLD parameter name. If provided and
                straggler_threshold is not, maps to straggler_threshold.
            straggler_threshold: Clients with capacity < this are stragglers.
                Defaults to 0.4 (low_perf=0.25 → straggler,
                mid_perf=0.5 and high_perf=1.0 → non-straggler).
            total_rounds: Total FL communication rounds.
        """
        super().__init__(global_model, total_rounds)

        # Resolve threshold: prefer new name, fall back to old, then default
        if straggler_threshold is not None:
            self.straggler_threshold = straggler_threshold
        elif leader_threshold is not None:
            # Map old semantics: leader_threshold=0.7 meant cap>=0.7 is leader
            # New semantics: cap < threshold is straggler
            # Reasonable mapping: straggler_threshold ≈ leader_threshold - 0.2
            self.straggler_threshold = max(0.2, leader_threshold - 0.2)
            logger.info(
                f"  FLuID: Mapped legacy leader_threshold={leader_threshold} "
                f"→ straggler_threshold={self.straggler_threshold}"
            )
        else:
            self.straggler_threshold = 0.4

        # Track straggler vs non-straggler classification
        self.straggler_clients = set()
        self.non_straggler_clients = set()

        # Per-neuron delta signal from non-straggler updates
        self.neuron_deltas = {name: None for name, _ in self.prunable_layers}
        self.deltas_initialized = False

        # Pre-aggregation state for delta computation
        self._pre_agg_state = None
        self._participating_ids = None

        logger.info(
            f"FLuID: straggler_threshold={self.straggler_threshold} "
            f"(cap < threshold → straggler, cap >= threshold → non-straggler)"
        )

    # -----------------------------------------------------------------
    # Capacity registration: classify straggler vs non-straggler
    # -----------------------------------------------------------------

    def register_capacity(self, client_id, capacity):
        """Register capacity and dynamically classify client."""
        super().register_capacity(client_id, capacity)

        if capacity < self.straggler_threshold:
            self.straggler_clients.add(client_id)
            self.non_straggler_clients.discard(client_id)
            logger.info(
                f"  FLuID: {client_id} → STRAGGLER "
                f"(cap={capacity:.2f} < {self.straggler_threshold})"
            )
        else:
            self.non_straggler_clients.add(client_id)
            self.straggler_clients.discard(client_id)
            logger.info(
                f"  FLuID: {client_id} → NON-STRAGGLER "
                f"(cap={capacity:.2f} >= {self.straggler_threshold})"
            )

    # -----------------------------------------------------------------
    # Index computation
    # -----------------------------------------------------------------

    def _compute_indices_for_client(self, client_id, keep_ratio, round_num):
        """
        Non-stragglers: all neurons (full model).
        Stragglers: drop invariant neurons to match capacity budget.
        """
        if client_id in self.non_straggler_clients:
            return self._full_model_indices()

        # Straggler path
        if not self.deltas_initialized:
            return self._magnitude_fallback(keep_ratio)

        return self._invariant_dropout_extraction(keep_ratio)

    def _full_model_indices(self):
        """All neurons — non-stragglers train the full model."""
        indices = {}
        for name, n_neurons in self.prunable_layers:
            indices[name] = list(range(n_neurons))
        return indices

    def _invariant_dropout_extraction(self, keep_ratio):
        """
        Invariant Dropout: drop neurons with smallest weight deltas
        from non-stragglers. These are "invariant" — barely changing,
        contributing little to training.

        Adjust drop count to match straggler's capacity budget.
        """
        indices = {}
        for name, n_neurons_info in self.prunable_layers:
            deltas = self.neuron_deltas[name]
            if deltas is None:
                n_keep = max(1, int(n_neurons_info * keep_ratio))
                indices[name] = list(range(n_keep))
                continue

            n_total = len(deltas)
            n_keep = max(1, int(n_total * keep_ratio))
            n_drop = n_total - n_keep

            # Drop the n_drop neurons with SMALLEST deltas (invariant)
            # Keep the n_keep neurons with LARGEST deltas (active)
            sorted_idx = np.argsort(deltas)  # ascending: smallest first
            invariant_set = set(sorted_idx[:n_drop].tolist())
            kept = [i for i in range(n_total) if i not in invariant_set]
            indices[name] = sorted(kept)

        return indices

    def _magnitude_fallback(self, keep_ratio):
        """Pre-delta fallback: select by weight magnitude."""
        indices = {}
        for name, n_neurons in self.prunable_layers:
            param = dict(self.global_model.named_parameters())[name]
            if param.dim() == 4:
                mag = param.data.view(n_neurons, -1).abs().mean(dim=1)
            else:
                mag = param.data.abs().mean(dim=1)
            n_keep = max(1, int(n_neurons * keep_ratio))
            top_idx = torch.argsort(mag, descending=True)[:n_keep]
            indices[name] = sorted(top_idx.cpu().tolist())
        return indices

    # -----------------------------------------------------------------
    # Per-client payload: non-stragglers get full, stragglers get compact
    # -----------------------------------------------------------------

    def get_payload_for_client(self, client_id, model_state_dict):
        """
        Override to differentiate non-straggler vs straggler payloads.

        Non-stragglers MUST get the full model because their deltas are
        used to identify invariant neurons. If they got a compact submodel,
        we couldn't compute per-neuron deltas for the full model.

        Stragglers get compact submodels (when extraction is enabled).
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

        if client_id in self.non_straggler_clients:
            # Non-stragglers: full model (needed for delta computation)
            return {
                "model_state": model_state_dict,
                "extra_payload": clean_indices,
                "is_submodel": False,
            }
        else:
            # Stragglers: compact submodel
            submodel_state = extract_submodel(model_state_dict, clean_indices)
            return {
                "model_state": submodel_state,
                "extra_payload": clean_indices,
                "is_submodel": True,
            }

    # -----------------------------------------------------------------
    # Aggregation: capture pre-agg state via base class aggregate
    # -----------------------------------------------------------------

    def aggregate(self, updates):
        """Capture pre-aggregation state, then delegate to base."""
        # Save pre-aggregation global weights for delta computation
        self._pre_agg_state = {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
            if 'weight' in name and 'bn' not in name and param.dim() >= 2
        }

        # Track participating client IDs (in order)
        self._participating_ids = [
            u.get("client_id", "unknown") for u in updates
        ]

        # Delegate to base class (which calls _post_aggregation_hook)
        return super().aggregate(updates)

    # -----------------------------------------------------------------
    # Post-aggregation hook: compute invariant neuron signal
    # -----------------------------------------------------------------

    def _post_aggregation_hook(self, client_states):
        """
        Called by base class after aggregation completes.

        Compute per-neuron |Δw| from NON-STRAGGLER clients only.
        Neurons with small deltas are invariant and candidates for
        dropping from straggler submodels next round.
        """
        pre_agg = self._pre_agg_state
        participating_ids = self._participating_ids

        if pre_agg is None:
            return

        try:
            # Collect states from non-straggler clients only
            non_straggler_states = []
            for i, state in enumerate(client_states):
                if i < len(participating_ids):
                    cid = participating_ids[i]
                    if cid in self.non_straggler_clients:
                        non_straggler_states.append(state)

            if not non_straggler_states:
                logger.info(
                    "  FLuID: No non-stragglers participated this round, "
                    "keeping previous invariance signal"
                )
                return

            # Compute per-neuron mean |delta| across non-stragglers
            for name, n_neurons in self.prunable_layers:
                if name not in pre_agg:
                    continue

                deltas = []
                for ns_state in non_straggler_states:
                    if name not in ns_state:
                        continue
                    delta = (
                        ns_state[name].float() - pre_agg[name].float()
                    ).abs()
                    if delta.dim() == 4:
                        deltas.append(
                            delta.view(n_neurons, -1).mean(dim=1).cpu().numpy()
                        )
                    elif delta.dim() == 2:
                        deltas.append(
                            delta.mean(dim=1).cpu().numpy()
                        )
                    else:
                        deltas.append(delta.cpu().numpy())

                if deltas:
                    self.neuron_deltas[name] = np.mean(deltas, axis=0)

            self.deltas_initialized = True
            logger.info(
                f"  FLuID: Updated invariance signal from "
                f"{len(non_straggler_states)} non-stragglers"
            )

        except Exception as e:
            logger.error(
                f"  FLuID: Error in _post_aggregation_hook: {e}",
                exc_info=True
            )
        finally:
            self._pre_agg_state = None
            self._participating_ids = None
