#!/usr/bin/env python
"""
Experiment Runner for Performance Benchmarking

This script automates the execution of serial and parallel training
experiments, collecting metrics for performance comparison.

Experiments:
1. Serial baseline training
2. Parallel training with varying process counts (2, 4, 8)
3. Speedup and efficiency calculations
4. Loss curve comparisons

Usage:
    python experiments/run_experiments.py --epochs 10 --process-counts 1 2 4 8
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.serial_trainer import train_serial
from src.training.parallel_trainer import train_parallel
from src.utils.metrics import TrainingResults
from src.utils.visualization import (
    plot_loss_curves,
    plot_speedup,
    plot_efficiency,
    plot_training_comparison
)


def run_serial_experiment(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_root: str,
    results_dir: Path
) -> TrainingResults:
    """
    Run the serial training experiment.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        data_root: Data root directory
        results_dir: Directory to save results
        
    Returns:
        TrainingResults from serial training
    """
    print("\n" + "=" * 70)
    print("RUNNING SERIAL BASELINE EXPERIMENT")
    print("=" * 70)
    
    results = train_serial(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        data_root=data_root,
        verbose=True
    )
    
    # Save results
    results_file = results_dir / "serial_results.json"
    results.save(str(results_file))
    print(f"Serial results saved to: {results_file}")
    
    return results


def run_parallel_experiment(
    world_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_root: str,
    results_dir: Path
) -> TrainingResults:
    """
    Run a parallel training experiment with specified world size.
    
    Args:
        world_size: Number of parallel processes
        epochs: Number of training epochs
        batch_size: Batch size per process
        learning_rate: Learning rate
        data_root: Data root directory
        results_dir: Directory to save results
        
    Returns:
        TrainingResults from parallel training
    """
    print("\n" + "=" * 70)
    print(f"RUNNING PARALLEL EXPERIMENT WITH {world_size} PROCESSES")
    print("=" * 70)
    
    results = train_parallel(
        world_size=world_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        data_root=data_root
    )
    
    # Save results
    results_file = results_dir / f"parallel_{world_size}p_results.json"
    results.save(str(results_file))
    print(f"Parallel ({world_size}p) results saved to: {results_file}")
    
    return results


def compute_performance_metrics(
    serial_results: TrainingResults,
    parallel_results: Dict[int, TrainingResults]
) -> Dict[str, Any]:
    """
    Compute speedup and efficiency metrics.
    
    Args:
        serial_results: Results from serial training
        parallel_results: Dictionary of results keyed by world_size
        
    Returns:
        Dictionary with performance metrics
    """
    serial_time = serial_results.total_training_time
    
    metrics = {
        'serial': {
            'time': serial_time,
            'throughput': sum(em.samples_per_second for em in serial_results.epoch_metrics) / len(serial_results.epoch_metrics),
            'final_accuracy': serial_results.final_val_accuracy
        },
        'parallel': {}
    }
    
    for world_size, results in parallel_results.items():
        parallel_time = results.total_training_time
        speedup = serial_time / parallel_time
        efficiency = speedup / world_size
        
        metrics['parallel'][world_size] = {
            'time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'throughput': sum(em.samples_per_second for em in results.epoch_metrics) / len(results.epoch_metrics),
            'final_accuracy': results.final_val_accuracy
        }
    
    return metrics


def print_summary(
    serial_results: TrainingResults,
    parallel_results: Dict[int, TrainingResults],
    performance_metrics: Dict[str, Any]
) -> None:
    """Print a summary of all experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print("\n1. TRAINING TIME COMPARISON")
    print("-" * 40)
    print(f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 40)
    
    serial_time = performance_metrics['serial']['time']
    print(f"{'Serial (1 process)':<20} {serial_time:<12.2f} {'1.00x':<10} {'100.0%':<10}")
    
    for world_size in sorted(performance_metrics['parallel'].keys()):
        pm = performance_metrics['parallel'][world_size]
        print(f"{'Parallel (' + str(world_size) + 'p)':<20} {pm['time']:<12.2f} {pm['speedup']:<10.2f}x {pm['efficiency']*100:<10.1f}%")
    
    print("\n2. ACCURACY COMPARISON")
    print("-" * 40)
    print(f"{'Configuration':<20} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 40)
    
    print(f"{'Serial':<20} {serial_results.final_train_accuracy:<12.4f} {serial_results.final_val_accuracy:<12.4f}")
    
    for world_size in sorted(parallel_results.keys()):
        results = parallel_results[world_size]
        print(f"{'Parallel (' + str(world_size) + 'p)':<20} {results.final_train_accuracy:<12.4f} {results.final_val_accuracy:<12.4f}")
    
    print("\n3. THROUGHPUT COMPARISON")
    print("-" * 40)
    print(f"{'Configuration':<20} {'Samples/sec':<15}")
    print("-" * 40)
    
    print(f"{'Serial':<20} {performance_metrics['serial']['throughput']:<15.0f}")
    
    for world_size in sorted(performance_metrics['parallel'].keys()):
        pm = performance_metrics['parallel'][world_size]
        print(f"{'Parallel (' + str(world_size) + 'p)':<20} {pm['throughput']:<15.0f}")
    
    print("\n" + "=" * 70)


def run_all_experiments(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    process_counts: List[int] = [2, 4],
    data_root: str = './data',
    output_dir: str = './results'
) -> Dict[str, Any]:
    """
    Run all experiments and generate comparison plots.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size (per process for parallel)
        learning_rate: Learning rate
        process_counts: List of process counts to test
        data_root: Data root directory
        output_dir: Output directory for results
        
    Returns:
        Dictionary with all results and metrics
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(output_dir) / f"experiment_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CNN PARALLEL TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Process counts: {process_counts}")
    print(f"  Results directory: {results_dir}")
    print(f"  Available CPU cores: {mp.cpu_count()}")
    
    # Run serial experiment
    serial_results = run_serial_experiment(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        data_root=data_root,
        results_dir=results_dir
    )
    
    # Run parallel experiments
    parallel_results = {}
    for world_size in process_counts:
        if world_size > mp.cpu_count():
            print(f"\nSkipping {world_size} processes (only {mp.cpu_count()} cores available)")
            continue
            
        parallel_results[world_size] = run_parallel_experiment(
            world_size=world_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            data_root=data_root,
            results_dir=results_dir
        )
    
    # Compute performance metrics
    performance_metrics = compute_performance_metrics(serial_results, parallel_results)
    
    # Save performance metrics
    metrics_file = results_dir / "performance_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(performance_metrics, f, indent=2)
    
    # Print summary
    print_summary(serial_results, parallel_results, performance_metrics)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    
    # Collect all results for plotting
    all_results = {'serial': serial_results}
    all_results.update({f'parallel_{k}p': v for k, v in parallel_results.items()})
    
    # Plot loss curves
    plot_loss_curves(
        all_results,
        save_path=str(results_dir / "loss_curves.png")
    )
    
    # Plot speedup
    speedups = {k: v['speedup'] for k, v in performance_metrics['parallel'].items()}
    plot_speedup(
        speedups,
        save_path=str(results_dir / "speedup.png")
    )
    
    # Plot efficiency
    efficiencies = {k: v['efficiency'] for k, v in performance_metrics['parallel'].items()}
    plot_efficiency(
        efficiencies,
        save_path=str(results_dir / "efficiency.png")
    )
    
    # Plot training comparison
    plot_training_comparison(
        serial_results,
        parallel_results,
        save_path=str(results_dir / "training_comparison.png")
    )
    
    print(f"\nPlots saved to: {results_dir}")
    
    return {
        'serial_results': serial_results,
        'parallel_results': parallel_results,
        'performance_metrics': performance_metrics,
        'results_dir': str(results_dir)
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run CNN parallel training experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size (per process for parallel)'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.01,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--process-counts', '-p',
        type=int,
        nargs='+',
        default=[2, 4],
        help='Process counts to test'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for MNIST dataset'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = run_all_experiments(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        process_counts=args.process_counts,
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    print("\nExperiment complete!")
    print(f"Results saved to: {results['results_dir']}")
