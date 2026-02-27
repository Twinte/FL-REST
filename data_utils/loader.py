import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import logging

logger = logging.getLogger(__name__)

def get_dataset(dataset_name, root_dir='./data', train=True, download=True):
    """Factory function to load different torchvision datasets."""
    dataset_name = dataset_name.upper()
    
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return torchvision.datasets.CIFAR10(root=root_dir, train=train, 
                                            download=download, transform=transform)
                                            
    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return torchvision.datasets.MNIST(root=root_dir, train=train, 
                                          download=download, transform=transform)
                                          
    elif dataset_name == "FASHIONMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return torchvision.datasets.FashionMNIST(root=root_dir, train=train, 
                                                 download=download, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

def prepare_client_loaders(dataset_name, total_clients, batch_size, alpha=None):
    """
    Splits the training dataset among clients.
    (Currently standard random split, can be expanded for Dirichlet/Non-IID here)
    """
    full_train_dataset = get_dataset(dataset_name, train=True)
    
    # Simple IID split for now
    split_size = len(full_train_dataset) // total_clients
    lengths = [split_size] * total_clients
    # Add remainder to the last client if it doesn't divide evenly
    lengths[-1] += len(full_train_dataset) - sum(lengths) 
    
    datasets = random_split(full_train_dataset, lengths)
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    
    return loaders