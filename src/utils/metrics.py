"""
Metrics Tracking Utilities

This module provides utilities for tracking and storing training metrics
such as loss, accuracy, and timing information.
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    epoch_time: float = 0.0  # seconds
    samples_per_second: float = 0.0


@dataclass
class TrainingResults:
    """Complete training results."""
    model_name: str
    training_type: str  # 'serial' or 'parallel'
    num_processes: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01
    num_epochs: int = 10
    total_training_time: float = 0.0
    final_train_loss: float = 0.0
    final_train_accuracy: float = 0.0
    final_val_loss: float = 0.0
    final_val_accuracy: float = 0.0
    epoch_metrics: List[EpochMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['epoch_metrics'] = [asdict(em) for em in self.epoch_metrics]
        return data
    
    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingResults':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        epoch_metrics = [EpochMetrics(**em) for em in data.pop('epoch_metrics')]
        results = cls(**data)
        results.epoch_metrics = epoch_metrics
        return results


class MetricsTracker:
    """
    Track training metrics during training.
    
    This class provides utilities for computing and storing metrics
    during the training process.
    """
    
    def __init__(
        self,
        model_name: str = "SimpleCNN",
        training_type: str = "serial",
        num_processes: int = 1,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        num_epochs: int = 10
    ):
        self.results = TrainingResults(
            model_name=model_name,
            training_type=training_type,
            num_processes=num_processes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        
        self._epoch_start_time: Optional[float] = None
        self._training_start_time: Optional[float] = None
        
        # Running metrics for current epoch
        self._running_loss = 0.0
        self._running_correct = 0
        self._running_total = 0
    
    def start_training(self) -> None:
        """Mark the start of training."""
        self._training_start_time = time.time()
    
    def end_training(self) -> None:
        """Mark the end of training and compute final metrics."""
        if self._training_start_time is not None:
            self.results.total_training_time = time.time() - self._training_start_time
        
        if self.results.epoch_metrics:
            last_epoch = self.results.epoch_metrics[-1]
            self.results.final_train_loss = last_epoch.train_loss
            self.results.final_train_accuracy = last_epoch.train_accuracy
            if last_epoch.val_loss is not None:
                self.results.final_val_loss = last_epoch.val_loss
            if last_epoch.val_accuracy is not None:
                self.results.final_val_accuracy = last_epoch.val_accuracy
    
    def start_epoch(self) -> None:
        """Mark the start of an epoch and reset running metrics."""
        self._epoch_start_time = time.time()
        self._running_loss = 0.0
        self._running_correct = 0
        self._running_total = 0
    
    def update_batch(
        self,
        loss: float,
        predictions: 'torch.Tensor',
        targets: 'torch.Tensor'
    ) -> None:
        """
        Update running metrics with batch results.
        
        Args:
            loss: Loss value for the batch
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels
        """
        import torch
        
        batch_size = targets.size(0)
        self._running_loss += loss * batch_size
        self._running_total += batch_size
        
        # Compute accuracy
        if predictions.dim() > 1:
            # Predictions are logits, get class indices
            pred_classes = predictions.argmax(dim=1)
        else:
            pred_classes = predictions
        
        self._running_correct += (pred_classes == targets).sum().item()
    
    def end_epoch(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        num_samples: Optional[int] = None
    ) -> EpochMetrics:
        """
        End the current epoch and record metrics.
        
        Args:
            epoch: Current epoch number (0-indexed)
            val_loss: Optional validation loss
            val_accuracy: Optional validation accuracy
            num_samples: Total samples processed (for distributed training)
            
        Returns:
            EpochMetrics for the completed epoch
        """
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0
        
        train_loss = self._running_loss / max(self._running_total, 1)
        train_accuracy = self._running_correct / max(self._running_total, 1)
        
        # Compute samples per second
        total_samples = num_samples if num_samples else self._running_total
        samples_per_second = total_samples / max(epoch_time, 1e-6)
        
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            epoch_time=epoch_time,
            samples_per_second=samples_per_second
        )
        
        self.results.epoch_metrics.append(metrics)
        return metrics
    
    def get_results(self) -> TrainingResults:
        """Get the complete training results."""
        return self.results
    
    def save_results(self, filepath: str) -> None:
        """Save results to file."""
        self.results.save(filepath)
    
    def print_epoch_summary(self, metrics: EpochMetrics) -> None:
        """Print a summary of epoch metrics."""
        print(f"Epoch {metrics.epoch + 1}: "
              f"Loss={metrics.train_loss:.4f}, "
              f"Acc={metrics.train_accuracy:.4f}, "
              f"Time={metrics.epoch_time:.2f}s, "
              f"Throughput={metrics.samples_per_second:.0f} samples/s")
        
        if metrics.val_loss is not None:
            print(f"          Val Loss={metrics.val_loss:.4f}, "
                  f"Val Acc={metrics.val_accuracy:.4f}")


def compute_accuracy(predictions: 'torch.Tensor', targets: 'torch.Tensor') -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Model predictions (logits or class indices)
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        pred_classes = predictions.argmax(dim=1)
    else:
        pred_classes = predictions
    
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    
    return correct / total if total > 0 else 0.0
