#!/usr/bin/env python3
"""
Data Preparation for FL-REST
==============================
Downloads the dataset (CIFAR-10 or CIFAR-100) and creates non-IID
client partitions using Dirichlet allocation.

Reads from environment / config.py:
  - DATASET_NAME: "CIFAR10" or "CIFAR100"
  - DIRICHLET_ALPHA: concentration parameter (lower = more non-IID)
  - TOTAL_CLIENTS: number of client partitions to create
  - RANDOM_SEED: for reproducibility

Output:
  - ./data/{dataset files}
  - ./data/partitions.json  (client_id -> list of sample indices)
"""

import os
import json
import numpy as np
import torchvision
import torchvision.transforms as transforms
import config


def partition_data_dirichlet(dataset, n_clients, alpha, seed=42):
    """
    Partition dataset indices among n_clients using Dirichlet distribution.
    
    Args:
        dataset: torchvision dataset with .targets attribute
        n_clients: number of client partitions
        alpha: Dirichlet concentration (lower = more heterogeneous)
        seed: random seed
    
    Returns:
        dict: {client_id_str: [sample_indices]}
    """
    np.random.seed(seed)
    
    targets = np.array(dataset.targets)
    n_classes = len(set(targets))
    
    client_indices = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        
        # Dirichlet proportions for this class
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Convert proportions to actual counts
        proportions = (proportions * len(class_indices)).astype(int)
        # Fix rounding: assign remainder to random clients
        remainder = len(class_indices) - proportions.sum()
        for i in range(abs(remainder)):
            idx = np.random.randint(n_clients)
            proportions[idx] += 1 if remainder > 0 else -1
        
        # Split indices according to proportions
        start = 0
        for i in range(n_clients):
            end = start + proportions[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end
    
    # Shuffle each client's indices
    for i in range(n_clients):
        np.random.shuffle(client_indices[i])
    
    # Convert to string-keyed dict (JSON keys must be strings)
    # Keys are 1-indexed to match client_001, client_002, etc.
    partitions = {str(i + 1): client_indices[i] for i in range(n_clients)}
    
    return partitions


def main():
    print(f"=== Data Preparation ===")
    print(f"Dataset:    {config.DATASET_NAME}")
    print(f"Clients:    {config.TOTAL_CLIENTS}")
    print(f"Alpha:      {config.DIRICHLET_ALPHA}")
    print(f"Seed:       {config.RANDOM_SEED}")
    
    # 1. Download dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if config.DATASET_NAME == "CIFAR100":
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        dataset_cls = torchvision.datasets.CIFAR10
    
    print(f"\nDownloading {config.DATASET_NAME}...")
    train_dataset = dataset_cls(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_cls(
        root='./data', train=False, download=True, transform=transform)
    
    n_classes = len(set(train_dataset.targets))
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Classes: {n_classes}")
    
    # 2. Create Dirichlet partitions
    print(f"\nCreating {config.TOTAL_CLIENTS} partitions (Dirichlet α={config.DIRICHLET_ALPHA})...")
    partitions = partition_data_dirichlet(
        train_dataset, 
        config.TOTAL_CLIENTS, 
        config.DIRICHLET_ALPHA,
        config.RANDOM_SEED
    )
    
    # 3. Save partitions
    partition_file = './data/partitions.json'
    with open(partition_file, 'w') as f:
        json.dump(partitions, f)
    
    # 4. Print summary
    print(f"\nPartition summary:")
    sizes = [len(v) for v in partitions.values()]
    print(f"  Samples per client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    
    # Class distribution per client (first 3 clients)
    targets = np.array(train_dataset.targets)
    for cid in list(partitions.keys())[:3]:
        client_targets = targets[partitions[cid]]
        unique, counts = np.unique(client_targets, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(f"  Client {cid}: {len(partitions[cid])} samples, "
              f"classes={len(unique)}/{n_classes}, "
              f"top class has {max(counts)} samples")
    
    print(f"\nPartitions saved to {partition_file}")
    print("=== Data preparation complete ===")


if __name__ == "__main__":
    main()