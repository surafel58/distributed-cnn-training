"""
Parallel Training Implementation with PyTorch DDP

This module implements distributed data parallel training using PyTorch's
DistributedDataParallel (DDP) for the CNN model on MNIST.

Key concepts:
- Data Parallelism: Each process trains on a different shard of the data
- Gradient All-Reduce: Gradients are averaged across all processes
- Synchronized Updates: All processes apply identical weight updates
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn import SimpleCNN
from src.data.dataset import get_distributed_mnist_dataloader, get_mnist_dataloader
from src.utils.metrics import MetricsTracker, TrainingResults, EpochMetrics


def setup_distributed(rank: int, world_size: int, backend: str = 'gloo') -> None:
    """
    Initialize the distributed environment.
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        backend: Distributed backend ('gloo' for CPU, 'nccl' for GPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()


class ParallelTrainer:
    """
    Parallel trainer using PyTorch DDP.
    
    This trainer implements distributed data parallel training where:
    1. Each process gets a shard of the training data
    2. Each process computes gradients on its local batch
    3. Gradients are all-reduced (averaged) across processes
    4. All processes apply identical weight updates
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        num_epochs: int = 10,
        device: str = 'cpu',
        data_root: str = './data',
        verbose: bool = True
    ):
        """
        Initialize the parallel trainer for a specific process.
        
        Args:
            rank: Rank of this process (0 to world_size-1)
            world_size: Total number of processes
            batch_size: Batch size per process
            learning_rate: Learning rate for SGD
            momentum: Momentum for SGD
            num_epochs: Number of training epochs
            device: Device to use ('cpu')
            data_root: Root directory for MNIST data
            verbose: Whether to print progress (only rank 0)
        """
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.data_root = data_root
        self.verbose = verbose and (rank == 0)  # Only rank 0 prints
        
        # Initialize model and wrap with DDP
        self.model = SimpleCNN().to(self.device)
        self.ddp_model = DDP(self.model)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.ddp_model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        
        # Distributed data loaders
        self.train_loader, self.train_sampler = get_distributed_mnist_dataloader(
            batch_size=batch_size,
            train=True,
            root=data_root,
            rank=rank,
            world_size=world_size
        )
        
        # Test loader (same for all processes)
        self.test_loader = get_mnist_dataloader(
            batch_size=batch_size,
            train=False,
            shuffle=False,
            root=data_root
        )
        
        # Metrics tracker (only rank 0 tracks)
        if self.rank == 0:
            self.metrics = MetricsTracker(
                model_name="SimpleCNN",
                training_type="parallel",
                num_processes=world_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
        else:
            self.metrics = None
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with distributed data parallel.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Tuple of (average loss, accuracy) for this process
        """
        self.ddp_model.train()
        
        # Set epoch for sampler (ensures different shuffling each epoch)
        self.train_sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        # Progress bar (only for rank 0)
        if self.verbose:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (through DDP model)
            output = self.ddp_model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass (DDP handles gradient synchronization)
            loss.backward()
            
            # Parameter update (synchronized across all processes)
            self.optimizer.step()
            
            # Track local metrics
            batch_size = target.size(0)
            running_loss += loss.item() * batch_size
            pred = output.argmax(dim=1)
            running_correct += (pred == target).sum().item()
            running_total += batch_size
            
            if self.verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute local averages
        avg_loss = running_loss / running_total
        accuracy = running_correct / running_total
        
        return avg_loss, accuracy, running_total
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        
        Only rank 0 performs full evaluation to avoid redundant computation.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.ddp_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.ddp_model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * target.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Optional[TrainingResults]:
        """
        Run the complete distributed training loop.
        
        Returns:
            TrainingResults object (only for rank 0)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Parallel Training - SimpleCNN on MNIST")
            print(f"{'='*60}")
            print(f"World size (processes): {self.world_size}")
            print(f"Batch size per process: {self.batch_size}")
            print(f"Effective batch size: {self.batch_size * self.world_size}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Momentum: {self.momentum}")
            print(f"Epochs: {self.num_epochs}")
            print(f"Device: {self.device}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Samples per process: ~{len(self.train_loader) * self.batch_size}")
            print(f"Model parameters: {self.model.get_num_params():,}")
            print(f"{'='*60}\n")
        
        # Synchronize all processes before starting
        dist.barrier()
        
        if self.rank == 0:
            self.metrics.start_training()
        
        training_start = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Start epoch timing for metrics (rank 0 only)
            if self.rank == 0:
                self.metrics.start_epoch()
            
            # Train for one epoch
            train_loss, train_acc, local_samples = self.train_epoch(epoch)
            
            # Aggregate metrics across all processes
            # Create tensors for all-reduce
            metrics_tensor = torch.tensor(
                [train_loss * local_samples, train_acc * local_samples, local_samples],
                dtype=torch.float64
            )
            
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            
            total_samples = int(metrics_tensor[2].item())
            global_loss = metrics_tensor[0].item() / total_samples
            global_acc = metrics_tensor[1].item() / total_samples
            
            epoch_time = time.time() - epoch_start
            
            # Evaluate on test set (all ranks for consistency)
            val_loss, val_acc = self.evaluate()
            
            # Record metrics (only rank 0)
            if self.rank == 0:
                self.metrics._running_loss = global_loss * total_samples
                self.metrics._running_correct = int(global_acc * total_samples)
                self.metrics._running_total = total_samples
                
                epoch_metrics = self.metrics.end_epoch(
                    epoch=epoch,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    num_samples=total_samples
                )
                
                if self.verbose:
                    self.metrics.print_epoch_summary(epoch_metrics)
            
            # Synchronize before next epoch
            dist.barrier()
        
        total_time = time.time() - training_start
        
        if self.rank == 0:
            self.metrics.end_training()
            self.metrics.results.total_training_time = total_time
            
            if self.verbose:
                results = self.metrics.get_results()
                print(f"\n{'='*60}")
                print(f"Training Complete!")
                print(f"Total time: {results.total_training_time:.2f}s")
                print(f"Final train accuracy: {results.final_train_accuracy:.4f}")
                print(f"Final test accuracy: {results.final_val_accuracy:.4f}")
                print(f"{'='*60}\n")
            
            return self.metrics.get_results()
        
        return None
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model (only rank 0)."""
        if self.rank == 0:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            # Save the underlying model, not the DDP wrapper
            torch.save(self.model.state_dict(), filepath)


def worker_process(
    rank: int,
    world_size: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    num_epochs: int,
    data_root: str,
    results_queue: mp.Queue
) -> None:
    """
    Worker function for each parallel process.
    
    Args:
        rank: Rank of this process
        world_size: Total number of processes
        batch_size: Batch size per process
        learning_rate: Learning rate
        momentum: Momentum
        num_epochs: Number of epochs
        data_root: Data root directory
        results_queue: Queue to send results back to main process
    """
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size)
        
        # Create trainer and run training
        trainer = ParallelTrainer(
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            num_epochs=num_epochs,
            data_root=data_root,
            verbose=True
        )
        
        results = trainer.train()
        
        # Only rank 0 sends results
        if rank == 0 and results is not None:
            results_queue.put(results.to_dict())
        
        # Cleanup
        cleanup_distributed()
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        if rank == 0:
            results_queue.put(None)


def train_parallel(
    world_size: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_epochs: int = 10,
    data_root: str = './data',
    save_path: Optional[str] = None
) -> TrainingResults:
    """
    Convenience function to run parallel training.
    
    This spawns multiple processes and coordinates distributed training.
    
    Args:
        world_size: Number of parallel processes
        batch_size: Batch size per process
        learning_rate: Learning rate for SGD
        momentum: Momentum for SGD
        num_epochs: Number of training epochs
        data_root: Root directory for MNIST data
        save_path: Optional path to save trained model
        
    Returns:
        TrainingResults object
    """
    # Use spawn method for Windows compatibility
    mp.set_start_method('spawn', force=True)
    
    # Queue for receiving results from rank 0
    results_queue = mp.Queue()
    
    # Spawn worker processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(
                rank,
                world_size,
                batch_size,
                learning_rate,
                momentum,
                num_epochs,
                data_root,
                results_queue
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Get results from queue
    results_dict = results_queue.get()
    
    if results_dict is None:
        raise RuntimeError("Training failed - no results returned")
    
    # Reconstruct TrainingResults from dict
    epoch_metrics = [EpochMetrics(**em) for em in results_dict.pop('epoch_metrics')]
    results = TrainingResults(**results_dict)
    results.epoch_metrics = epoch_metrics
    
    if save_path:
        results.save(save_path.replace('.pt', '_results.json'))
    
    return results


if __name__ == "__main__":
    # Quick test with 2 processes
    print("Testing parallel training with 2 processes...")
    
    results = train_parallel(
        world_size=2,
        batch_size=64,
        learning_rate=0.01,
        num_epochs=2,
        data_root='./data'
    )
    
    print(f"\nTest Results:")
    print(f"Total time: {results.total_training_time:.2f}s")
    print(f"Final accuracy: {results.final_val_accuracy:.4f}")
