"""
Serial Training Implementation

This module implements the serial (single-process) training baseline
for the CNN model on MNIST. This serves as the reference implementation
for comparing parallel training performance.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cnn import SimpleCNN
from src.data.dataset import get_mnist_dataloader
from src.utils.metrics import MetricsTracker, compute_accuracy, TrainingResults


class SerialTrainer:
    """
    Serial trainer for CNN on MNIST.
    
    This trainer implements standard single-process training with
    forward pass, loss computation, backward pass, and parameter updates.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        num_epochs: int = 10,
        device: str = 'cpu',
        data_root: str = './data',
        verbose: bool = True
    ):
        """
        Initialize the serial trainer.
        
        Args:
            model: PyTorch model to train (creates SimpleCNN if None)
            batch_size: Number of samples per batch
            learning_rate: Learning rate for SGD optimizer
            momentum: Momentum for SGD optimizer
            num_epochs: Number of training epochs
            device: Device to train on ('cpu' or 'cuda')
            data_root: Root directory for MNIST data
            verbose: Whether to print training progress
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.data_root = data_root
        self.verbose = verbose
        
        # Initialize model
        self.model = model if model is not None else SimpleCNN()
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        
        # Data loaders
        self.train_loader = get_mnist_dataloader(
            batch_size=batch_size,
            train=True,
            shuffle=True,
            root=data_root
        )
        self.test_loader = get_mnist_dataloader(
            batch_size=batch_size,
            train=False,
            shuffle=False,
            root=data_root
        )
        
        # Metrics tracker
        self.metrics = MetricsTracker(
            model_name="SimpleCNN",
            training_type="serial",
            num_processes=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        self.metrics.start_epoch()
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            disable=not self.verbose
        )
        
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output, target)
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Parameter update
            self.optimizer.step()
            
            # Update metrics
            self.metrics.update_batch(loss.item(), output.detach(), target)
            
            # Update progress bar
            if self.verbose:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        
        return self.metrics._running_loss / self.metrics._running_total, \
               self.metrics._running_correct / self.metrics._running_total
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * target.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> TrainingResults:
        """
        Run the complete training loop.
        
        Returns:
            TrainingResults object with all metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Serial Training - SimpleCNN on MNIST")
            print(f"{'='*60}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Momentum: {self.momentum}")
            print(f"Epochs: {self.num_epochs}")
            print(f"Device: {self.device}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Test samples: {len(self.test_loader.dataset)}")
            print(f"Model parameters: {self.model.get_num_params():,}")
            print(f"{'='*60}\n")
        
        self.metrics.start_training()
        
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate on test set
            val_loss, val_acc = self.evaluate()
            
            # Record epoch metrics
            epoch_metrics = self.metrics.end_epoch(
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_acc,
                num_samples=len(self.train_loader.dataset)
            )
            
            if self.verbose:
                self.metrics.print_epoch_summary(epoch_metrics)
        
        self.metrics.end_training()
        
        if self.verbose:
            results = self.metrics.get_results()
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Total time: {results.total_training_time:.2f}s")
            print(f"Final train accuracy: {results.final_train_accuracy:.4f}")
            print(f"Final test accuracy: {results.final_val_accuracy:.4f}")
            print(f"{'='*60}\n")
        
        return self.metrics.get_results()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))


def train_serial(
    batch_size: int = 64,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    num_epochs: int = 10,
    device: str = 'cpu',
    data_root: str = './data',
    verbose: bool = True,
    save_path: Optional[str] = None
) -> TrainingResults:
    """
    Convenience function to run serial training.
    
    Args:
        batch_size: Number of samples per batch
        learning_rate: Learning rate for SGD
        momentum: Momentum for SGD
        num_epochs: Number of training epochs
        device: Device to train on
        data_root: Root directory for MNIST data
        verbose: Whether to print progress
        save_path: Optional path to save trained model
        
    Returns:
        TrainingResults object
    """
    trainer = SerialTrainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        num_epochs=num_epochs,
        device=device,
        data_root=data_root,
        verbose=verbose
    )
    
    results = trainer.train()
    
    if save_path:
        trainer.save_model(save_path)
        results.save(save_path.replace('.pt', '_results.json'))
    
    return results


if __name__ == "__main__":
    # Run a quick test
    results = train_serial(
        batch_size=64,
        learning_rate=0.01,
        num_epochs=2,
        verbose=True
    )
    
    print(f"\nTest Results:")
    print(f"Total time: {results.total_training_time:.2f}s")
    print(f"Final accuracy: {results.final_val_accuracy:.4f}")
