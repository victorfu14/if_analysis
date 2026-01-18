import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataset(cfg):
    """
    Returns: dataset, input_dim, output_dim
    """
    if cfg.data.dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
        input_dim = 784
        output_dim = 10

    elif cfg.data.dataset_name == "digits":
        # 8x8 Digits Dataset (dimension = 64) ---
        digits = load_digits()
        X = digits.data  # Shape: (1797, 64)
        y = digits.target
        
        # Standardize (Critical for neural nets)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Wrap in TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        input_dim = 64
        output_dim = 10

    elif cfg.data.dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root=cfg.data.root, train=True, download=True, transform=transform)
        input_dim = 3072
        output_dim = 10
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")
        
    return dataset, input_dim, output_dim

# def get_dataset(cfg):
#     """Loads the raw dataset object (without the loader)."""
#     if cfg.data.dataset_name == "mnist":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         dataset = datasets.MNIST(root=cfg.data.root, train=True, download=True, transform=transform)
#         input_dim = 784
#         output_dim = 10
#     elif cfg.data.dataset_name == "cifar10":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         dataset = datasets.CIFAR10(root=cfg.data.root, train=True, download=True, transform=transform)
#         input_dim = 3072
#         output_dim = 10
#     else:
#         raise ValueError("Unknown dataset")
        
#     return dataset, input_dim, output_dim

def get_loo_loader(dataset, exclude_index=None, batch_size=32):
    """
    Returns a DataLoader. 
    If exclude_index is provided, that specific sample is removed.
    """
    total_len = len(dataset)
    indices = list(range(total_len))
    
    if exclude_index is not None:
        # Remove the specific index
        indices.pop(exclude_index)
    
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=True, # Shuffle training data
        num_workers=0
    )
    return loader