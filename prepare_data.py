import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import config
import logging

from data_utils.loader import get_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data():
    logger.info(f"--- Starting Data Preparation for {config.DATASET_NAME} ---")
    
    # 1. Download Dataset using the factory
    logger.info(f"Checking/Downloading {config.DATASET_NAME} dataset...")
    
    train_dataset = get_dataset(config.DATASET_NAME, train=True, download=True)
    test_dataset = get_dataset(config.DATASET_NAME, train=False, download=True)

    # 2. Calculate Non-IID Partitions
    logger.info(f"Partitioning data for {config.TOTAL_CLIENTS} clients (Alpha={config.DIRICHLET_ALPHA})...")
    
    np.random.seed(config.RANDOM_SEED)
    
    # Dynamically get the number of classes from the dataset
    num_classes = len(train_dataset.classes)
    labels = np.array(train_dataset.targets)
    
    # Get indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_partitions = [[] for _ in range(config.TOTAL_CLIENTS)]

    # Dirichlet distribution logic
    for k in range(num_classes):
        img_idx_k = class_indices[k]
        np.random.shuffle(img_idx_k)
        
        # Generate proportions
        proportions = np.random.dirichlet(np.repeat(config.DIRICHLET_ALPHA, config.TOTAL_CLIENTS))
        
        # Calculate split points
        proportions_cumsum = (np.cumsum(proportions) * len(img_idx_k)).astype(int)[:-1]
        
        # Split and assign
        split_indices = np.split(img_idx_k, proportions_cumsum)
        for i in range(config.TOTAL_CLIENTS):
            client_partitions[i].extend(split_indices[i].tolist())

    # 3. Save partitions to JSON
    partition_file = './data/partitions.json'
    logger.info(f"Saving partitions to {partition_file}...")
    
    # Convert to a dictionary: "1": [indices], "2": [indices]
    partition_dict = {
        str(i + 1): indices 
        for i, indices in enumerate(client_partitions)
    }
    
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)
    
    with open(partition_file, 'w') as f:
        json.dump(partition_dict, f)
        
    logger.info("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()