import torch
import torch.distributed as dist

print(f"PyTorch version: {torch.__version__}")
print(f"Gloo available: {dist.is_gloo_available()}")
print(f"NCCL available: {dist.is_nccl_available()}")
