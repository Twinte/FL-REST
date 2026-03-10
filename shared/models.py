import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 (3 channels).
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleMLP(nn.Module):
    """
    A simple MLP for flattened inputs.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512) # CIFAR is 32x32x3
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FedPruneCNN(nn.Module):
    """
    Larger CNN for partial training / FedPrune experiments.
    
    Architecture: 3 conv blocks (32→64→128 filters) with BatchNorm
    and 3 FC layers (256→128→num_classes). Provides enough neurons 
    per layer for meaningful importance-based pruning.
    
    Supports both CIFAR-10 (num_classes=10) and CIFAR-100 (num_classes=100).
    
    Note: BatchNorm layers are NOT pruned — they are always averaged
    across all clients. Only conv weight and fc weight layers are 
    subject to neuron selection.
    """
    def __init__(self, num_classes=10):
        super(FedPruneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FedPruneCNN_Wide(nn.Module):
    """
    Wider CNN for CIFAR-100 partial training experiments.
    
    Same conv backbone as FedPruneCNN but with wider FC layers:
      fc1: 2048 → 512  (was 256)
      fc2: 512 → 256   (was 128)
      fc3: 256 → 100
    
    This gives 512+256 = 768 FC neurons (vs 256+128 = 384 in the base).
    At capacity 0.5, a client trains 128 FC neurons for 100 classes —
    enough representational capacity to avoid the bottleneck.
    
    Total prunable neurons: 32+64+128+512+256 = 992 (vs 608 in base).
    """
    def __init__(self, num_classes=100):
        super(FedPruneCNN_Wide, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- Model Registry ---
def get_model(model_name):
    """Factory function to instantiate models by name."""
    if model_name == "SimpleCNN":
        return SimpleCNN()
    elif model_name == "SimpleMLP":
        return SimpleMLP()
    elif model_name == "FedPruneCNN":
        return FedPruneCNN(num_classes=10)
    elif model_name == "FedPruneCNN100":
        return FedPruneCNN_Wide(num_classes=100)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")