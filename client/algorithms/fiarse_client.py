"""
FIARSE Client Algorithm for FL-REST
=====================================
Client-side gradient-based importance estimation
(Anonymous, "FIARSE: Model-Heterogeneous FL via Importance-Aware
Submodel Extraction", NeurIPS 2024)

Protocol:
  1. Receive model + neuron indices from server
  2. Compute grad x weight importance on local data (EXTRA COST)
  3. Train with gradient masking (same as other partial-training methods)
  4. Upload model update + importance scores

The importance computation is the key differentiator and extra burden:
  - Requires a full-model forward+backward pass on local data
  - This is the "weak client paradox" -- devices too weak for full
    training must still run full-model gradient computation
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
    FIARSE: computes client-side importance then trains with masking.
    
    The importance dict is returned in the training result under
    key "client_importance" -- the existing tensor_metrics pipeline
    in client/app.py automatically sends it to the server.
    """
    
    def __init__(self, importance_batches=5):
        self.importance_batches = importance_batches
    
    def train(self, model, dataloader, device, epochs, global_c=None):
        """
        1. Compute importance (extra forward+backward pass)
        2. Train with gradient masking
        3. Return model update + importance scores
        """
        neuron_indices = global_c
        
        model.to(device)
        
        # ============================================================
        # STEP 1: Compute grad x weight importance (FIARSE extra cost)
        # ============================================================
        client_importance = self._compute_importance(model, dataloader, device)
        
        logger.info(
            f"FIARSE: Computed importance for "
            f"{sum(len(v) for v in client_importance.values())} neurons "
            f"across {len(client_importance)} layers "
            f"({self.importance_batches} batches)"
        )
        
        # ============================================================
        # STEP 2: Train with gradient masking (identical to others)
        # ============================================================
        model.train()
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM
        )
        criterion = nn.CrossEntropyLoss()
        
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
                f"FIARSE: Training {total_assigned} neurons across "
                f"{len(neuron_indices)} layers"
            )
        else:
            logger.warning("FIARSE: No neuron indices -- training full model")
        
        total_loss = 0.0
        total_batches = 0
        
        for epoch in range(epochs):
            for data, labels in dataloader:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        if param.dim() == 4:
                            param.grad *= masks[name].view(-1, 1, 1, 1)
                        elif param.dim() == 2:
                            param.grad *= masks[name].view(-1, 1)
                        elif param.dim() == 1:
                            param.grad *= masks[name]
                
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / max(total_batches, 1)
        
        return {
            "algorithm": "FIARSE",
            "avg_loss": avg_loss,
            "neurons_trained": sum(len(v) for v in neuron_indices.values()) if neuron_indices else 0,
            "layers_masked": len(masks),
            # Picked up by client/app.py tensor_metrics detection,
            # piggybacked to server alongside the model update
            "client_importance": client_importance,
        }
    
    def _compute_importance(self, model, dataloader, device):
        """
        Per-neuron importance via |grad x weight| (first-order Taylor).
        
        Requires a full-model forward+backward pass over local data.
        
        Returns:
            dict: {layer_name: torch.Tensor of shape [n_neurons]}
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        importance = {}
        for name, param in model.named_parameters():
            if 'weight' in name and 'bn' not in name and param.dim() >= 2:
                importance[name] = torch.zeros(param.size(0), device=device)
        
        batch_count = 0
        for data, target in dataloader:
            if batch_count >= self.importance_batches:
                break
            
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if name in importance and param.grad is not None:
                    gw = (param.grad * param.data).abs()
                    if param.dim() == 4:
                        importance[name] += gw.view(param.size(0), -1).sum(dim=1)
                    elif param.dim() == 2:
                        importance[name] += gw.sum(dim=1)
            
            batch_count += 1
        
        if batch_count > 0:
            for name in importance:
                importance[name] = importance[name] / batch_count
        
        return importance
