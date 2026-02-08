# H200 Cache Thrashing Fix

## Problem Identified
Your H200 training run had **164,025 trace cache invalidations** vs only 6 on H100, causing identical performance despite superior hardware (141GB vs 80GB memory).

## Root Cause
PyTorch's compilation cache is constantly being invalidated on H200, likely due to:
- Dynamic tensor shapes during GRPO training
- Different GPU architecture requiring different compilation strategies
- ZeRO-3's memory allocation patterns conflicting with trace compilation

## Solutions (Apply in order)

### 1. Modify DeepSpeed Config (CRITICAL)
Edit `/arf/scratch/aalatan/FewShotReasoning/train/local_scripts/zero3.json`:

**Replace "auto" values with fixed sizes:**

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 500000000,
        "stage3_prefetch_bucket_size": 50000000,
        "stage3_param_persistence_threshold": 100000,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "stage3_use_all_reduce_for_fetch_params": false
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

### 2. Add Environment Variables to Job Script
Add these BEFORE launching training:

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_COMPILE_DISABLE="1"
export CUDA_LAUNCH_BLOCKING="0"
export NCCL_IB_DISABLE="0"
export NCCL_P2P_DISABLE="0"
```

### 3. Create Pre-Training Setup Script
Create `/arf/scratch/aalatan/FewShotReasoning/train/setup_h200.py`:

```python
import torch
import os

# Disable problematic PyTorch features
if hasattr(torch._dynamo, 'config'):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 2048  # Increase cache

# Configure CUDA memory allocator
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Disable torch.compile if causing issues
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

print("✅ H200 optimization settings applied")
```

Then import this at the top of `grpo.py`:
```python
import setup_h200  # Add this line at the top
```

### 4. Modify GRPO Training Script (Optional but Recommended)
Add gradient checkpointing settings to reduce memory pressure.

In your GRPO config, ensure:
```python
training_args = GRPOConfig(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Add this
    ...
)
```

### 5. Verify Cache Flushes Are Fixed
After applying fixes, check the .err file for:
- **Before**: 164,025 "Invalidate trace cache" messages
- **After**: Should be < 100 messages

Look for this warning:
```
[WARNING] [stage3.py:2114:step] XX pytorch allocator cache flushes since last step
```
If XX > 5, you still have memory pressure issues.

## Expected Performance Improvement
- **Current**: H200 = H100 (~23 hours)
- **After fix**: H200 should be 1.3-1.5x faster (~15-18 hours)
- Reduced memory pressure warnings
- Stable cache performance

## Testing the Fix
1. Apply changes above
2. Run a short test (1-2 steps): `--max_steps 2`
3. Check `.err` file for cache invalidation messages
4. If < 100 messages, run full training

## Alternative: Increase Micro Batch Size (If memory allows)
Since H200 has 141GB, you can try:
```json
"train_micro_batch_size_per_gpu": 2  // Instead of 1
```
This reduces the number of forward/backward passes and can help with cache stability.
