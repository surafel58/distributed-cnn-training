"""
MNIST Dataset Loading Utilities

This module provides utilities for loading the MNIST dataset for both
serial and distributed (parallel) training scenarios.

Key features:
- Standard dataloader for serial training
- Distributed dataloader with DistributedSampler for parallel training
- Consistent preprocessing and normalization
"""

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os


# MNIST normalization values (precomputed)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms() -> transforms.Compose:
    """
    Get the standard transforms for MNIST.
    
    Returns:
        Composed transforms for preprocessing MNIST images
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])


def get_mnist_dataset(
    root: str = './data',
    train: bool = True,
    download: bool = True
) -> datasets.MNIST:
    """
    Load the MNIST dataset.
    
    Args:
        root: Root directory for storing the dataset
        train: If True, load training set; otherwise load test set
        download: If True, download the dataset if not present
        
    Returns:
        MNIST dataset object
    """
    return datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=get_transforms()
    )


def get_mnist_dataloader(
    batch_size: int = 64,
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    root: str = './data'
) -> DataLoader:
    """
    Create a standard DataLoader for MNIST (serial training).
    
    Args:
        batch_size: Number of samples per batch
        train: If True, load training set; otherwise load test set
        shuffle: If True, shuffle the data at every epoch
        num_workers: Number of subprocesses for data loading
        pin_memory: If True, pin memory for faster GPU transfer
        root: Root directory for the dataset
        
    Returns:
        DataLoader for MNIST dataset
    """
    dataset = get_mnist_dataset(root=root, train=train)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train  # Drop last incomplete batch during training
    )


def get_distributed_mnist_dataloader(
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    root: str = './data',
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create a distributed DataLoader for MNIST (parallel training with DDP).
    
    The DistributedSampler ensures each process gets a unique, non-overlapping
    subset of the data. This is essential for data parallelism.
    
    Args:
        batch_size: Number of samples per batch (per process)
        train: If True, load training set; otherwise load test set
        num_workers: Number of subprocesses for data loading
        pin_memory: If True, pin memory for faster GPU transfer
        root: Root directory for the dataset
        rank: Rank of the current process
        world_size: Total number of processes
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (DataLoader, DistributedSampler or None)
        The sampler is returned so it can be updated each epoch
    """
    dataset = get_mnist_dataset(root=root, train=train)
    
    if train:
        # Use DistributedSampler for training
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True
        )
        shuffle = False  # Sampler handles shuffling
    else:
        # For evaluation, we can use regular sampling
        # Each process will evaluate the full test set
        sampler = None
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=train
    )
    
    return dataloader, sampler


def get_dataset_info(root: str = './data') -> dict:
    """
    Get information about the MNIST dataset.
    
    Args:
        root: Root directory for the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    train_dataset = get_mnist_dataset(root=root, train=True)
    test_dataset = get_mnist_dataset(root=root, train=False)
    
    return {
        'name': 'MNIST',
        'num_train_samples': len(train_dataset),
        'num_test_samples': len(test_dataset),
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'mean': MNIST_MEAN,
        'std': MNIST_STD
    }


if __name__ == "__main__":
    # Test the dataloaders
    print("Testing MNIST Data Loading Utilities")
    print("=" * 50)
    
    # Test serial dataloader
    train_loader = get_mnist_dataloader(batch_size=64, train=True)
    test_loader = get_mnist_dataloader(batch_size=64, train=False)
    
    print(f"\nSerial Training DataLoader:")
    print(f"  Number of batches: {len(train_loader)}")
    print(f"  Batch size: 64")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"  Sample batch - Images: {images.shape}, Labels: {labels.shape}")
    
    print(f"\nSerial Test DataLoader:")
    print(f"  Number of batches: {len(test_loader)}")
    
    # Test distributed dataloader (simulated with world_size=2)
    print("\n" + "=" * 50)
    print("Testing Distributed DataLoader (simulated)")
    
    for rank in range(2):
        dist_loader, sampler = get_distributed_mnist_dataloader(
            batch_size=64,
            train=True,
            rank=rank,
            world_size=2
        )
        print(f"\nRank {rank}:")
        print(f"  Number of batches: {len(dist_loader)}")
        print(f"  Samples per process: ~{len(dist_loader) * 64}")
    
    # Dataset info
    print("\n" + "=" * 50)
    info = get_dataset_info()
    print("\nDataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
