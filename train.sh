#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=grpo_cls_500_500
#SBATCH --output=grpo-%j.out
#SBATCH --error=grpo-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH -C H100

export PATH="/arf/home/aalatan/mert/envs/train2/bin:$PATH"
export CONDA_PREFIX="/arf/home/aalatan/mert/envs/train2"
export CONDA_DEFAULT_ENV="train2"
export LD_LIBRARY_PATH="/arf/home/aalatan/mert/envs/train2/lib:$LD_LIBRARY_PATH"

cd /arf/scratch/aalatan/FewShotReasoning/train
chmod +x /arf/home/aalatan/mert/envs/train2/lib/python3.10/site-packages/wandb/bin/wandb-core

export PYTHONPATH=/arf/scratch/aalatan/FewShotReasoning/train/src:/arf/home/aalatan/mert/envs/train2/lib/python3.10/site-packages:$PYTHONPATH
export WANDB_RUN_NAME=Qwen-VL-2B-GRPO-500Easy-500Hard-Temp1.5-2epoch-cosine$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_MODE=offline
export WANDB_DIR=/arf/scratch/aalatan/FewShotReasoning/train/wandb
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export TQDM_MININTERVAL=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p $WANDB_DIR

echo "========== Job Info =========="
echo "Run: $WANDB_RUN_NAME"
echo "Dataset: 500 Easy / 500 Hard"
echo "Strategy: Temp 1.5, Cosine, 2 Epochs, Min Pixels Fixed"
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
    --model_name_or_path /arf/scratch/aalatan/Qwen2-VL-2B-CLS-CoT \
    --dataset_name /arf/scratch/aalatan/VHM_dataset_grpo_cls_balanced_500_500 \
    --max_prompt_length 8192 \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.001 \
    --report_to wandb \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 153664 \
    --max_pixels 1204224 \
    --save_total_limit 15 \
    --num_train_epochs 2 \
    --num_generations 8 \
    --save_steps 10 \
    --run_name $WANDB_RUN_NAME \
    --disable_tqdm false \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2

echo "Training completed!"
echo "WandB logs saved to: $WANDB_DIR"