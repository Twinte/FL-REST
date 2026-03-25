"""
FIARSE Client Algorithm for FL-REST (CORRECTED)
==================================================
Threshold-Controlled Biased Gradient Descent (TCB-GD).
(Wu et al., NeurIPS 2024)

PAPER ALGORITHM (Algorithm 1, Lines 4-11):
  1. Receive submodel from server
  2. For K local iterations, apply TCB-GD (Equation 3):
     g = ∇F_i(x ⊙ M(x)) ⊙ M(x) ⊙ (1 + 2|x|θ / (|x|+θ)²)
  3. Send delta back to server

KEY PROPERTIES:
  - Threshold-controlled: only parameters ≥ threshold are updated.
  - Biased: the bias term accelerates separation of important vs
    unimportant parameters near the threshold boundary.
  - No separate importance computation — zero extra client overhead.
  - Mask can dynamically shrink as parameters drop below threshold.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import config
from .base import ClientAlgorithm

logger = logging.getLogger(__name__)


class FIARSEClient(ClientAlgorithm):
    """
    FIARSE: Trains with Threshold-Controlled Biased Gradient Descent.
    """

    def __init__(self, importance_batches=5):
        """
        Args:
            importance_batches: Accepted for factory compatibility but NOT USED.
                The old (wrong) implementation used this for gradient-based
                importance computation. Real FIARSE has no separate importance
                step — TCB-GD IS the training algorithm.
        """
        # importance_batches intentionally unused
        pass

    def train(self, model, dataloader, device, epochs, global_c=None):
        """
        TCB-GD training with threshold-controlled biased gradients.

        Args:
            model: PyTorch model (full or submodel-loaded).
            dataloader: Client's local data.
            device: torch device.
            epochs: Number of local training epochs.
            global_c: Neuron indices dict from server.
                      Format: {layer_name: [int, int, ...]}

        Returns:
            dict: Training metrics. NO client_importance field.
        """
        neuron_indices = global_c
        model.to(device)
        model.train()

        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM
        )
        criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------
        # Compute per-neuron threshold θ for TCB-GD bias term.
        # θ = magnitude of the smallest assigned neuron in each layer
        # (approximates TopK_γ threshold from the paper)
        # ---------------------------------------------------------
        thresholds = {}
        if neuron_indices is not None:
            for name, param in model.named_parameters():
                if name in neuron_indices and param.dim() >= 2:
                    idx_list = neuron_indices[name]
                    if isinstance(idx_list, torch.Tensor):
                        idx_list = idx_list.tolist()
                    if param.dim() == 4:
                        neuron_mags = param.data.abs().view(
                            param.size(0), -1).mean(dim=1)
                    else:
                        neuron_mags = param.data.abs().mean(dim=1)
                    assigned_mags = neuron_mags[idx_list]
                    thresholds[name] = assigned_mags.min().item()

        # Build neuron-level masks for gradient masking
        masks = {}
        if neuron_indices is not None:
            for name, param in model.named_parameters():
                if name in neuron_indices:
                    mask = torch.zeros(param.size(0), device=device)
                    idx_list = neuron_indices[name]
                    if isinstance(idx_list, torch.Tensor):
                        idx_list = idx_list.tolist()
                    for idx in idx_list:
                        mask[idx] = 1.0
                    masks[name] = mask

            total_assigned = sum(len(v) for v in neuron_indices.values())
            logger.info(
                f"FIARSE-TCB: Training {total_assigned} neurons across "
                f"{len(neuron_indices)} layers"
            )
        else:
            logger.warning("FIARSE-TCB: No neuron indices — training full model")

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()

                # ---------------------------------------------------
                # Apply TCB-GD: mask + bias term (Equation 3)
                #
                # g = ∇F(x ⊙ M) ⊙ M ⊙ (1 + 2|x|θ / (|x| + θ)²)
                # ---------------------------------------------------
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        # Step 1: Apply neuron mask (threshold-controlled)
                        m = masks[name]
                        if param.dim() == 4:
                            shaped_mask = m.view(-1, 1, 1, 1)
                        elif param.dim() == 2:
                            shaped_mask = m.view(-1, 1)
                        else:
                            shaped_mask = m

                        param.grad *= shaped_mask

                        # Step 2: Apply TCB-GD bias term per-parameter
                        if name in thresholds:
                            theta = thresholds[name]
                            if theta > 0:
                                abs_w = param.data.abs()
                                bias = 1.0 + (2.0 * abs_w * theta) / (
                                    (abs_w + theta) ** 2 + 1e-10
                                )
                                param.grad *= bias

                optimizer.step()

                # ---------------------------------------------------
                # Dynamic mask shrinkage: neurons that dropped below
                # threshold get masked out (paper Section 4.1)
                # Check periodically to avoid per-batch overhead.
                # ---------------------------------------------------
                if neuron_indices is not None and total_batches % 50 == 0:
                    for name, param in model.named_parameters():
                        if name in masks and name in thresholds:
                            theta = thresholds[name]
                            if param.dim() == 4:
                                neuron_mags = param.data.abs().view(
                                    param.size(0), -1).mean(dim=1)
                            elif param.dim() == 2:
                                neuron_mags = param.data.abs().mean(dim=1)
                            else:
                                neuron_mags = param.data.abs()
                            below = neuron_mags < theta
                            masks[name] = masks[name] * (~below).float()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)

        return {
            "algorithm": "FIARSE",
            "avg_loss": avg_loss,
            "neurons_trained": sum(
                len(v) for v in neuron_indices.values()
            ) if neuron_indices else 0,
            "layers_masked": len(masks),
            # NO client_importance — that's the whole point of corrected FIARSE
        }
