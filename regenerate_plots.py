"""Regenerate plots at higher quality from saved results."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.metrics import TrainingResults, EpochMetrics
from src.utils.visualization import plot_loss_curves, plot_speedup, plot_efficiency, plot_training_comparison

def load_results(filepath):
    """Load TrainingResults from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct TrainingResults
    results = TrainingResults(
        model_name=data['model_name'],
        training_type=data['training_type'],
        num_processes=data.get('num_processes', 1),
        batch_size=data.get('batch_size', 64),
        total_training_time=data['total_training_time'],
        final_train_accuracy=data['final_train_accuracy'],
        final_val_accuracy=data['final_val_accuracy']
    )
    
    # Reconstruct epoch metrics
    for em_data in data['epoch_metrics']:
        em = EpochMetrics(
            epoch=em_data['epoch'],
            train_loss=em_data['train_loss'],
            train_accuracy=em_data['train_accuracy'],
            val_loss=em_data.get('val_loss'),
            val_accuracy=em_data.get('val_accuracy'),
            epoch_time=em_data.get('epoch_time', 0),
            samples_per_second=em_data.get('samples_per_second', 0)
        )
        results.epoch_metrics.append(em)
    
    return results

def main():
    # Find the results directory
    results_dir = Path("results/experiment_20260202_021927")
    output_dir = Path(".")
    
    print("Loading results...")
    
    # Load serial results
    serial_results = load_results(results_dir / "serial_results.json")
    print(f"  Loaded serial results: {serial_results.total_training_time:.2f}s")
    
    # Load parallel results
    parallel_results = {}
    for p in [2, 4, 8, 12]:
        filepath = results_dir / f"parallel_{p}p_results.json"
        if filepath.exists():
            parallel_results[p] = load_results(filepath)
            print(f"  Loaded parallel {p}p results: {parallel_results[p].total_training_time:.2f}s")
    
    print("\nRegenerating plots at 300 DPI...")
    
    # Prepare loss curves data
    all_results = {"Serial": serial_results}
    for p in sorted(parallel_results.keys()):
        all_results[f"Parallel ({p}p)"] = parallel_results[p]
    
    # Generate loss curves
    plot_loss_curves(all_results, save_path=str(output_dir / "loss_curves.png"))
    
    # Calculate speedup and efficiency
    serial_time = serial_results.total_training_time
    speedups = {}
    efficiencies = {}
    for p, r in parallel_results.items():
        speedups[p] = serial_time / r.total_training_time
        efficiencies[p] = speedups[p] / p
    
    # Generate speedup plot
    plot_speedup(speedups, save_path=str(output_dir / "speedup.png"))
    
    # Generate efficiency plot
    plot_efficiency(efficiencies, save_path=str(output_dir / "efficiency.png"))
    
    # Generate training comparison
    plot_training_comparison(serial_results, parallel_results, 
                            save_path=str(output_dir / "training_comparison.png"))
    
    print("\nDone! High-quality plots saved to current directory.")

if __name__ == "__main__":
    main()
