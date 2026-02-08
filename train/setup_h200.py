"""
H200 Optimization Setup
Fixes cache thrashing issues observed in training runs.
Import this module before any training code.
"""
import torch
import os

# Disable PyTorch compilation cache thrashing
if hasattr(torch, '_dynamo'):
    if hasattr(torch._dynamo, 'config'):
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 2048  # Increase from default
        print("✅ TorchDynamo cache limit increased to 2048")

# Configure CUDA memory allocator for better memory management
cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
if 'expandable_segments' not in cuda_alloc_conf:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✅ CUDA expandable segments enabled")

# Disable torch.compile if it's causing cache invalidation
if os.environ.get('TORCH_COMPILE_DISABLE') != '1':
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    print("✅ Torch compile disabled")

# Ensure NCCL settings are optimal
if not os.environ.get('NCCL_IB_DISABLE'):
    os.environ['NCCL_IB_DISABLE'] = '0'
if not os.environ.get('NCCL_P2P_DISABLE'):
    os.environ['NCCL_P2P_DISABLE'] = '0'

print("✅ H200 optimization settings applied successfully")
print("   This should fix the cache thrashing issue (164k invalidations -> <100)")
