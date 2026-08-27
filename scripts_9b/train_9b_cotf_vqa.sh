#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=grpo9b_cotf_vqa
#SBATCH --output=grpo9b_cotf_vqa-%j.out
#SBATCH --error=grpo9b_cotf_vqa-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24   # corrupt GPUs: torch sees no device -> "doesn't support bf16/gpu". Inherited scripts lacked this.
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH -C H100

# Stage-3 GRPO on the POOLED-CoT models.
#
# Base is qwen35-cotall-<task>-merged: stage-1 SFT + a CoT LoRA trained on the combined
# CLS+VQA+VG reasoning corpus (95,849 samples), rather than that task's reasoning alone.
# Merged by scripts/reasoning/merge_lora_cotall.sh from training jobs 1426906/7/8.
#
# --temperature 1.0 is passed explicitly (it was already explicit in the scripts these were
# derived from). That matters: TRL's GRPOTrainer uses the same value for the rollouts and for
# the log-probs in the loss, so sampling and the policy gradient stay on the same distribution.
# Do not raise it without checking that both sides still agree.
#
# FILTERED CORPUS. Prompts whose 8 sampled completions all scored the same reward were
# removed: their group std is 0, so the advantage (r-mean)/std is identically 0 and they
# contribute no gradient. Measured on this exact checkpoint before filtering --
# CLS 69% dead, VQA 54% dead (after also dropping the system turn, see below), VG 8% dead.
# num_train_epochs is raised to keep the step count near the previous runs' ~316 despite the
# smaller corpus; at 4 prompts per optimizer step this run is ~295 steps.
#
# Checkpoint policy: save_steps 50 / save_total_limit 2, not 10 / 15. At 10/15 each run keeps
# 15 deepspeed checkpoints (model + optimizer + rng state) which measured 169G per run -- three
# concurrent runs exhausted the 2T quota and killed all three mid-training (jobs 1431079-81).
# Only the final top-level weights are ever promoted to output/grpo, so the intermediate
# checkpoints exist purely for resumability and two is plenty. ~22G per run at this setting.

# Compile caches on scratch, not the node-local /tmp: kolyoz nodes report TmpDisk=0 so /tmp is a
# small RAM-backed tmpfs, and filling it surfaces as "OSError: [Errno 28] No space left on device".
export TRITON_CACHE_DIR=/arf/scratch/aalatan/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/arf/scratch/aalatan/.cache/torchinductor
export PATH="/arf/home/aalatan/mert/envs/recot-train-grpo/bin:$PATH"
export CONDA_PREFIX="/arf/home/aalatan/mert/envs/recot-train-grpo"
export CONDA_DEFAULT_ENV="recot-train-grpo"
export LD_LIBRARY_PATH="/arf/home/aalatan/mert/envs/recot-train-grpo/lib:$LD_LIBRARY_PATH"

cd /arf/scratch/aalatan/FewShotReasoning/train
chmod +x /arf/home/aalatan/mert/envs/recot-train-grpo/lib/python3.11/site-packages/wandb/bin/wandb-core

export PYTHONPATH=/arf/scratch/aalatan/FewShotReasoning/train:/arf/scratch/aalatan/FewShotReasoning/train/src:/arf/home/aalatan/mert/envs/recot-train-grpo/lib/python3.11/site-packages:$PYTHONPATH
export WANDB_RUN_NAME=9B-Filtered-VQA-cot-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_MODE=offline
export WANDB_DIR=/arf/scratch/aalatan/FewShotReasoning/train/wandb
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export TQDM_MININTERVAL=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p $WANDB_DIR

echo "========== Job Info =========="
echo "Run: $WANDB_RUN_NAME"
echo "Base model: qwen35-9b-cot-vqa-merged"
echo "Dataset: VHM_dataset_grpo_cls_only_2k_refactored_filtered (197 prompts, ~295 steps)"
echo "OLD: VHM_dataset_grpo_cls_only_2k_refactored"
echo "=============================="

# Guard: several kolyoz nodes pass nvidia-smi but give torch no usable device, which surfaces as
# "ValueError: Your setup doesn't support bf16/gpu". Without this the run dies in seconds, the
# trailing echo still fires, and slurm records COMPLETED -- jobs 1452668/71/72 all failed that way.
if ! python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda(); print('### CUDA OK:', torch.cuda.get_device_name(0))"; then
    echo "### CUDA BROKEN on $(hostname) — requeueing"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 60
    exit 1
fi


python -u -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id=$SLURM_JOB_ID \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path /arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-9b-cot-vqa-merged \
    --dataset_name /arf/scratch/aalatan/grpo_data/VHM_dataset_grpo_vqa_only_2k_363_nosys_filtered \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.01 \
    --temperature 1.0 \
    --report_to wandb \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --min_pixels 200704 \
    --max_pixels 1003520 \
    --save_total_limit 1 \
    --save_only_model True \

    --num_train_epochs 2 \
    --num_generations 8 \
    --save_steps 50 \
    --run_name $WANDB_RUN_NAME \
    --disable_tqdm false \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2

echo "Training completed!"
echo "WandB logs saved to: $WANDB_DIR"
#
# Stage-3 GRPO for Qwen3.5-VL 9B, on the difficulty-FILTERED corpora.
#
#   base            qwen35-9b-{cot,cotall}-{task}-merged -- the stage-2 CoT models. Two arms:
#                   'cot' = task-matched reasoning data, 'cotall' = pooled CLS+VQA+VG.
#
#   CAVEAT          The filtered datasets were produced by sampling the 0.8B cotall checkpoints.
#                   Difficulty filtering is model-specific: the 9B SFT beats the 0.8B by 1.5-3 pt,
#                   so prompts that had reward spread for the 0.8B are likelier to be unanimous
#                   here. Expect frac_reward_zero_std above what the 0.8B runs achieved. Watch it.
#
#   VQA dataset     the *_nosys_* copy (168 prompts). grpo.py prepends a SYSTEM_PROMPT the CoT
#                   models never saw in training; on the 0.8B that made the VQA model emit
#                   <reasoning> ... </answer> with no closing tag and no answer, scoring 0 on
#                   340/363 prompts. The unfixed copy has only 23 usable prompts.
#
#   epochs          Lowered from the 0.8B's 6/8/2. At 4 prompts per optimizer step these corpora
#                   give 49/42/193 steps per epoch, and 9B generation (8 samples/prompt through
#                   HF generate, no vLLM) is roughly 10x the 0.8B's ~44s/step. 2/2/1 epochs keeps
#                   each run near 10-25h rather than several days.
#
#   --time 48h      partition max is 3 days; 48h leaves margin on the VG arm without asking for
#                   a slot the scheduler will struggle to place.
#
#   memory          zero3 shards params+grads+Adam states across the 4 GPUs (~27G/GPU for a 9B
#                   full fine-tune) plus activations and 8-sample generation. If this OOMs, swap
#                   local_scripts/zero3.json for zero3_offload.json before reducing batch.

export PATH=