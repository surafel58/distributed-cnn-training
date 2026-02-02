"""
Visualization Utilities

This module provides plotting utilities for visualizing training results,
including loss curves, speedup plots, and efficiency comparisons.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.metrics import TrainingResults


def setup_plot_style():
    """Configure matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8


def plot_loss_curves(
    results: Dict[str, TrainingResults],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot training loss curves for multiple training runs.
    
    Args:
        results: Dictionary mapping run names to TrainingResults
        save_path: Optional path to save the plot
        show: Whether to display the plot interactively
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        epochs = [em.epoch + 1 for em in result.epoch_metrics]
        train_losses = [em.train_loss for em in result.epoch_metrics]
        train_accs = [em.train_accuracy for em in result.epoch_metrics]
        
        # Plot loss
        ax1.plot(epochs, train_losses, label=name, color=color, marker='o')
        
        # Plot accuracy
        ax2.plot(epochs, train_accs, label=name, color=color, marker='o')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Loss curves saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_speedup(
    speedups: Dict[int, float],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot speedup vs number of processes.
    
    Args:
        speedups: Dictionary mapping process count to speedup value
        save_path: Optional path to save the plot
        show: Whether to display the plot interactively
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by process count
    process_counts = sorted(speedups.keys())
    speedup_values = [speedups[p] for p in process_counts]
    
    # Add serial baseline (1 process = 1x speedup)
    all_counts = [1] + process_counts
    all_speedups = [1.0] + speedup_values
    
    # Plot actual speedup
    ax.plot(all_counts, all_speedups, 'b-o', label='Actual Speedup', linewidth=2, markersize=10)
    
    # Plot ideal (linear) speedup
    ideal_speedups = all_counts
    ax.plot(all_counts, ideal_speedups, 'r--', label='Ideal Speedup', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title('Parallel Training Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0, max(all_counts) + 0.5)
    ax.set_ylim(0, max(max(all_speedups), max(ideal_speedups)) * 1.1)
    
    # Set x-ticks to show all process counts
    ax.set_xticks(all_counts)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Speedup plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_efficiency(
    efficiencies: Dict[int, float],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot parallel efficiency vs number of processes.
    
    Args:
        efficiencies: Dictionary mapping process count to efficiency value
        save_path: Optional path to save the plot
        show: Whether to display the plot interactively
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by process count
    process_counts = sorted(efficiencies.keys())
    efficiency_values = [efficiencies[p] * 100 for p in process_counts]  # Convert to percentage
    
    # Add serial baseline (1 process = 100% efficiency)
    all_counts = [1] + process_counts
    all_efficiencies = [100.0] + efficiency_values
    
    # Plot efficiency
    ax.bar(all_counts, all_efficiencies, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add ideal efficiency line
    ax.axhline(y=100, color='r', linestyle='--', label='Ideal Efficiency (100%)', linewidth=2)
    
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Training Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set axis limits
    ax.set_ylim(0, 110)
    ax.set_xticks(all_counts)
    
    # Add value labels on bars
    for i, (count, eff) in enumerate(zip(all_counts, all_efficiencies)):
        ax.text(count, eff + 2, f'{eff:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Efficiency plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_training_comparison(
    serial_results: TrainingResults,
    parallel_results: Dict[int, TrainingResults],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Create a comprehensive comparison plot of training results.
    
    Args:
        serial_results: Results from serial training
        parallel_results: Dictionary of parallel results keyed by world_size
        save_path: Optional path to save the plot
        show: Whether to display the plot interactively
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    configs = ['Serial'] + [f'{p}p Parallel' for p in sorted(parallel_results.keys())]
    times = [serial_results.total_training_time] + \
            [parallel_results[p].total_training_time for p in sorted(parallel_results.keys())]
    accuracies = [serial_results.final_val_accuracy] + \
                 [parallel_results[p].final_val_accuracy for p in sorted(parallel_results.keys())]
    
    # Compute throughputs
    serial_throughput = sum(em.samples_per_second for em in serial_results.epoch_metrics) / len(serial_results.epoch_metrics)
    throughputs = [serial_throughput]
    for p in sorted(parallel_results.keys()):
        r = parallel_results[p]
        throughputs.append(sum(em.samples_per_second for em in r.epoch_metrics) / len(r.epoch_metrics))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
    
    # Plot 1: Training Time
    ax1 = axes[0, 0]
    bars1 = ax1.bar(configs, times, color=colors, edgecolor='black')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, time in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Test Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.bar(configs, [a * 100 for a in accuracies], color=colors, edgecolor='black')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Final Test Accuracy Comparison')
    ax2.set_ylim(min(accuracies) * 100 - 2, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Throughput
    ax3 = axes[1, 0]
    bars3 = ax3.bar(configs, throughputs, color=colors, edgecolor='black')
    ax3.set_ylabel('Throughput (samples/second)')
    ax3.set_title('Training Throughput Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, tp in zip(bars3, throughputs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{tp:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Loss Curves
    ax4 = axes[1, 1]
    
    # Serial loss curve
    serial_epochs = [em.epoch + 1 for em in serial_results.epoch_metrics]
    serial_losses = [em.train_loss for em in serial_results.epoch_metrics]
    ax4.plot(serial_epochs, serial_losses, 'o-', label='Serial', linewidth=2)
    
    # Parallel loss curves
    for p in sorted(parallel_results.keys()):
        r = parallel_results[p]
        epochs = [em.epoch + 1 for em in r.epoch_metrics]
        losses = [em.train_loss for em in r.epoch_metrics]
        ax4.plot(epochs, losses, 'o-', label=f'{p}p Parallel', linewidth=2)
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training Loss')
    ax4.set_title('Training Loss Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Training comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_epoch_times(
    results: Dict[str, TrainingResults],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot per-epoch training times.
    
    Args:
        results: Dictionary mapping run names to TrainingResults
        save_path: Optional path to save the plot
        show: Whether to display the plot interactively
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        epochs = [em.epoch + 1 for em in result.epoch_metrics]
        times = [em.epoch_time for em in result.epoch_metrics]
        ax.plot(epochs, times, label=name, color=color, marker='o')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Per-Epoch Training Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Epoch times plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Demo with sample data
    print("Visualization utilities loaded successfully.")
    print("\nAvailable functions:")
    print("  - plot_loss_curves(results, save_path)")
    print("  - plot_speedup(speedups, save_path)")
    print("  - plot_efficiency(efficiencies, save_path)")
    print("  - plot_training_comparison(serial_results, parallel_results, save_path)")
    print("  - plot_epoch_times(results, save_path)")
