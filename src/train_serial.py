#!/usr/bin/env python
"""
Serial Training Entry Point

This script provides a command-line interface for running serial
(single-process) training of the CNN on MNIST.

Usage:
    python src/train_serial.py --epochs 10 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.serial_trainer import train_serial


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Serial training for CNN on MNIST',
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
        help='Batch size for training'
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
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
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
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for serial training."""
    args = parse_args()
    
    # Run training
    results = train_serial(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        num_epochs=args.epochs,
        device=args.device,
        data_root=args.data_root,
        verbose=not args.quiet,
        save_path=args.save_model
    )
    
    # Save results if requested
    if args.save_results:
        results.save(args.save_results)
        if not args.quiet:
            print(f"Results saved to: {args.save_results}")
    
    return results


if __name__ == "__main__":
    main()
