# CNN Parallel Training on MNIST

This project implements and compares serial and parallel training of a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch's Distributed Data Parallel (DDP).

## Project Overview

This assignment demonstrates:

1. **Serial Baseline**: Standard single-process training loop
2. **Parallel Training**: Distributed Data Parallel (DDP) training with multiple CPU processes
3. **Performance Analysis**: Speedup, efficiency, and convergence comparisons

### Architecture

The project uses a simple CNN architecture:

```
Conv2d(1, 32, 3) → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3) → ReLU → MaxPool2d(2)
Flatten → Linear(3136, 128) → ReLU → Dropout → Linear(128, 10)
```

### Parallelization Strategy

**Data Parallelism with PyTorch DDP:**

- Dataset is partitioned across processes using `DistributedSampler`
- Each process computes gradients on its local batch
- Gradients are synchronized via all-reduce (averaged across processes)
- All processes apply identical weight updates

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- numpy 1.24+
- matplotlib 3.7+
- tqdm 4.65+

## Installation

1. Clone or download this project

2. Create a virtual environment (recommended):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
Project/
├── src/
│   ├── models/
│   │   └── cnn.py              # CNN architecture
│   ├── data/
│   │   └── dataset.py          # MNIST data loading
│   ├── training/
│   │   ├── serial_trainer.py   # Serial training
│   │   └── parallel_trainer.py # DDP parallel training
│   ├── utils/
│   │   ├── metrics.py          # Metrics tracking
│   │   └── visualization.py    # Plotting utilities
│   ├── train_serial.py         # Serial training CLI
│   └── train_parallel.py       # Parallel training CLI
├── experiments/
│   └── run_experiments.py      # Benchmarking script
├── results/                    # Output directory
├── requirements.txt
└── README.md
```

## Usage

### 1. Serial Training

Run the serial baseline:

```bash
python src/train_serial.py --epochs 10 --batch-size 64
```

Options:

- `--epochs`, `-e`: Number of training epochs (default: 10)
- `--batch-size`, `-b`: Batch size (default: 64)
- `--learning-rate`, `-lr`: Learning rate (default: 0.01)
- `--save-results`: Path to save results JSON

### 2. Parallel Training

Run parallel training with multiple processes:

```bash
python src/train_parallel.py --world-size 4 --epochs 10 --batch-size 64
```

Options:

- `--world-size`, `-w`: Number of parallel processes (default: 2)
- `--epochs`, `-e`: Number of epochs (default: 10)
- `--batch-size`, `-b`: Batch size per process (default: 64)
- `--save-results`: Path to save results JSON

### 3. Run Full Experiments

Run the complete benchmarking suite:

```bash
python experiments/run_experiments.py --epochs 10 --process-counts 2 4 8
```

This will:

1. Run serial training baseline
2. Run parallel training with 2, 4, and 8 processes
3. Compute speedup and efficiency metrics
4. Generate comparison plots
5. Save all results to the `results/` directory

Options:

- `--epochs`, `-e`: Number of epochs (default: 10)
- `--batch-size`, `-b`: Batch size (default: 64)
- `--process-counts`, `-p`: List of process counts to test (default: 2 4)
- `--output-dir`, `-o`: Output directory (default: ./results)

## Expected Results

### Training Configuration

| Parameter                | Value              |
| ------------------------ | ------------------ |
| Batch size (per process) | 64                 |
| Learning rate            | 0.01               |
| Optimizer                | SGD (momentum=0.9) |
| Loss function            | CrossEntropyLoss   |
| Epochs                   | 10                 |

### Performance Metrics

The experiments will generate:

1. **Speedup**: `T_serial / T_parallel`
2. **Efficiency**: `Speedup / N_processes`
3. **Throughput**: Samples processed per second
4. **Accuracy**: Final test set accuracy

### Output Files

After running experiments, you'll find in `results/experiment_<timestamp>/`:

- `serial_results.json`: Serial training metrics
- `parallel_2p_results.json`: Parallel training with 2 processes
- `parallel_4p_results.json`: Parallel training with 4 processes
- `performance_metrics.json`: Speedup and efficiency calculations
- `loss_curves.png`: Training loss comparison
- `speedup.png`: Speedup vs. process count
- `efficiency.png`: Parallel efficiency
- `training_comparison.png`: Comprehensive comparison

## Technical Details

### Data Parallelism in DDP

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Data (MNIST)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌─────────┐     ┌─────────┐     ┌─────────┐
        │Process 0│     │Process 1│     │Process N│
        │ Shard 0 │     │ Shard 1 │     │ Shard N │
        └────┬────┘     └────┬────┘     └────┬────┘
             │               │               │
             ▼               ▼               ▼
        Forward Pass    Forward Pass    Forward Pass
             │               │               │
             ▼               ▼               ▼
        Local Grads     Local Grads     Local Grads
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                    ┌─────────────────┐
                    │   All-Reduce    │
                    │ (Average Grads) │
                    └─────────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
        optimizer.step() optimizer.step() optimizer.step()
        (Synchronized)   (Synchronized)   (Synchronized)
```

### Communication Backend

- **Gloo**: Used for CPU-based distributed training
- Supports all-reduce, broadcast, and other collective operations
- Optimized for multi-process communication on a single machine

### Gradient Synchronization

DDP automatically synchronizes gradients during the backward pass:

1. Each process computes local gradients
2. Gradients are all-reduced (sum + divide by world_size)
3. All processes have identical averaged gradients
4. Weight updates are identical across all processes

## Performance Considerations

### Challenges and Mitigations

| Challenge              | Cause                               | Mitigation                          |
| ---------------------- | ----------------------------------- | ----------------------------------- |
| Communication overhead | Gradient synchronization            | Gradient compression, async updates |
| Load imbalance         | Uneven data distribution            | DistributedSampler with shuffling   |
| Startup cost           | Process spawning                    | Amortize over many epochs           |
| Effective batch size   | Larger batches may hurt convergence | Learning rate scaling               |

### Tips for Better Performance

1. **Increase epochs** for larger process counts to amortize startup costs
2. **Scale learning rate** with world size (linear scaling rule)
3. **Use more workers** for data loading if I/O bound
4. **Profile communication** to identify bottlenecks

## Troubleshooting

### Common Issues

1. **"Address already in use" error**
   - Change the `MASTER_PORT` in `parallel_trainer.py`
   - Wait a few seconds before rerunning

2. **Out of memory**
   - Reduce batch size
   - Reduce number of parallel processes

3. **Slow performance with many processes**
   - Communication overhead dominates for small batch sizes
   - Increase batch size per process

4. **Training diverges with parallel training**
   - Ensure all processes use the same random seed
   - Check that gradients are being synchronized correctly

## License

This project is for educational purposes as part of a parallel programming course assignment.

## References

- [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed Data Parallel in PyTorch](https://pytorch.org/docs/stable/notes/ddp.html)
- [Data Parallelism](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
