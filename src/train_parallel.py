#!/usr/bin/env python
"""
Parallel Training Entry Point

This script provides a command-line interface for running distributed
data parallel training of the CNN on MNIST using PyTorch DDP.

Usage:
    python src/train_parallel.py --world-size 4 --epochs 10 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.parallel_trainer import train_parallel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parallel training for CNN on MNIST using DDP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--world-size', '-w',
        type=int,
        default=2,
        help='Number of parallel processes'
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
        help='Batch size per process'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.01,
        help='Learning rate for SGD optimizer'
    )
    
    parser.add_argument(
        '--momentum', '-m',
        type=float,
        default=0.9,
        help='Momentum for SGD optimizer'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='./data',
        help='Root directory for MNIST dataset'
    )
    
    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save training results (JSON)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for parallel training."""
    args = parse_args()
    
    print(f"Starting parallel training with {args.world_size} processes...")
    
    # Run training
    results = train_parallel(
        world_size=args.world_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_epochs=args.epochs,
        data_root=args.data_root,
        save_path=args.save_model
    )
    
    # Save results if requested
    if args.save_results:
        results.save(args.save_results)
        print(f"Results saved to: {args.save_results}")
    
    return results


if __name__ == "__main__":
    main()
