"""
Partial Training Client Algorithm for FL-REST
===============================================
Trains only the neurons assigned by the server using gradient masking.

Used by ALL partial-training methods:
  - FedPrune (importance-based indices from server)
  - HeteroFL (first-k indices from server)
  - FedRolex (rolling window indices from server)

The client doesn't know or care how indices were computed — it just
receives them and masks gradients accordingly. This is the key insight
that lets us compare methods fairly: identical client code, different
server-side extraction logic.

The server sends neuron indices as the 'extra_payload' in the composite
model download. The trainer.py passes this as the `global_c` kwarg.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import config
from .base import ClientAlgorithm

logger = logging.getLogger(__name__)


class PartialTrainingClient(ClientAlgorithm):
    """
    Client-side partial training using server-assigned neuron indices.
    
    Gradient masking:
      - Forward pass uses the FULL model (all neurons compute activations)
      - Backward pass zeros out gradients for non-assigned neurons
      - Only assigned neurons get weight updates
    """
    
    def __init__(self, method_name="PartialTraining"):
        self.method_name = method_name
    
    def train(self, model, dataloader, device, epochs, global_c=None):
        """
        Train the model with gradient masking on assigned neurons.
        
        Args:
            model: PyTorch model to train.
            dataloader: Client's local data.
            device: torch.device (CPU or CUDA).
            epochs: Number of local training epochs.
            global_c: Neuron indices dict from server.
                      Format: {layer_name: [int, int, ...]}
                      If None, trains all neurons (full model fallback).
        
        Returns:
            dict: Training metrics.
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
        
        # --- Build gradient masks ---
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
                f"{self.method_name}: Training {total_assigned} neurons across "
                f"{len(neuron_indices)} layers"
            )
        else:
            logger.warning(
                f"{self.method_name}: No neuron indices received — training full model"
            )
        
        # --- Training loop with gradient masking ---
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Apply gradient masks
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        if param.dim() == 4:      # Conv: [out, in, h, w]
                            param.grad *= masks[name].view(-1, 1, 1, 1)
                        elif param.dim() == 2:    # Linear: [out, in]
                            param.grad *= masks[name].view(-1, 1)
                        elif param.dim() == 1:    # Bias: [out]
                            param.grad *= masks[name]
                
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        
        return {
            "algorithm": self.method_name,
            "avg_loss": avg_loss,
            "neurons_trained": sum(len(v) for v in neuron_indices.values()) if neuron_indices else 0,
            "layers_masked": len(masks),
        }