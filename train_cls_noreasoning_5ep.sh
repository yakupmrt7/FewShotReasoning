#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=grpo_cls_noreasoning_5ep_qwen35
#SBATCH --output=grpo_cls_noreasoning_5ep-%j.out
#SBATCH --error=grpo_cls_noreasoning_5ep-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz19,kolyoz24
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH -C H100

export PATH="/arf/home/aalatan/mert/envs/recot-train-grpo/bin:$PATH"
export CONDA_PREFIX="/arf/home/aalatan/mert/envs/recot-train-grpo"
export CONDA_DEFAULT_ENV="recot-train-grpo"
export LD_LIBRARY_PATH="/arf/home/aalatan/mert/envs/recot-train-grpo/lib:$LD_LIBRARY_PATH"

cd /arf/scratch/aalatan/FewShotReasoning/train
chmod +x /arf/home/aalatan/mert/envs/recot-train-grpo/lib/python3.11/site-packages/wandb/bin/wandb-core

export PYTHONPATH=/arf/scratch/aalatan/FewShotReasoning/train:/arf/scratch/aalatan/FewShotReasoning/train/src:/arf/home/aalatan/mert/envs/recot-train-grpo/lib/python3.11/site-packages:$PYTHONPATH
export WANDB_RUN_NAME=Qwen3.5-VL-CLS-NoReasoning-5ep-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_MODE=offline
export WANDB_DIR=/arf/scratch/aalatan/FewShotReasoning/train/wandb
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29512
export TQDM_MININTERVAL=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p $WANDB_DIR

echo "========== Job Info =========="
echo "Run: $WANDB_RUN_NAME"
echo "Deliberate no-reasoning ablation (user-requested): skip CoT entirely, GRPO directly from"
echo "the SFT-only checkpoint, using bare direct-answer prompts (no reasoning instruction, no"
echo "system prompt) and --reward_funcs accuracy only (no format reward)."
echo "Base model: qwen35-cls-merged (Qwen3.5-VL, SFT-only on CLS, no CoT LoRA)"
echo "Dataset: VHM_dataset_grpo_cls_noreasoning (same 634 examples as the original CLS GRPO run,"
echo "reverse-parsed to strip the reasoning instruction)"
echo "Same stabilized hyperparams as every other CLS GRPO run (beta=0.01, lr=1e-6, warmup=0.1)"
echo "=============================="

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
    --model_name_or_path /arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cls-merged \
    --dataset_name /arf/scratch/aalatan/VHM_dataset_grpo_cls_noreasoning \
    --reward_funcs accuracy \
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
    --save_total_limit 15 \
    --num_train_epochs 5 \
    --num_generations 8 \
    --save_steps 10 \
    --run_name $WANDB_RUN_NAME \
    --disable_tqdm false \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2

echo "Training completed!"
echo "WandB logs saved to: $WANDB_DIR"
