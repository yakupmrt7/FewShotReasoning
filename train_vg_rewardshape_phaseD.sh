#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=grpo_vg_rewardshape_phaseD_qwen35
#SBATCH --output=grpo_vg_rewardshape_phaseD-%j.out
#SBATCH --error=grpo_vg_rewardshape_phaseD-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
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
export WANDB_RUN_NAME=Qwen3.5-VL-VG-RewardShape-PhaseD-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_MODE=offline
export WANDB_DIR=/arf/scratch/aalatan/FewShotReasoning/train/wandb
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29506
export TQDM_MININTERVAL=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

mkdir -p $WANDB_DIR

echo "========== Job Info =========="
echo "Run: $WANDB_RUN_NAME"
echo "Phase D on top of Phase C: same reward (accuracy_shaped: format-gated, dense IoU +"
echo "threshold bonus at 0.5) and same base/dataset as the RewardShape run, but with Fable5's"
echo "Phase D training-config changes (still on the same 843-example set -- Phase B data"
echo "expansion not done yet, so epochs/dataset are kept as-is; only the training-dynamics"
echo "knobs change):"
echo "  - learning_rate 1e-6 -> 2e-6 (policy needs to travel further than the KL ball allowed)"
echo "  - beta 0.01 -> 0.003 (anchor loosened, not removed -- safe now because of the two"
echo "    structural guards below, unlike the earlier beta=0.001 run which collapsed with"
echo "    neither guard in place)"
echo "  - sync_ref_model=True, ref_model_sync_steps=200 (KL constrains step size against a"
echo "    moving reference, not total displacement from the original CoT checkpoint)"
echo "  - epsilon=0.2 / epsilon_high=0.28 (clip-higher: protects low-probability exploratory"
echo "    tokens, the structural fix for the earlier beta=0.001 entropy collapse)"
echo "  - gradient_accumulation_steps 8 -> 16 (halves cross-prompt gradient noise: 8"
echo "    prompts/step instead of 4)"
echo "Base model: qwen35-cot-vg-merged (same as original + RewardShape runs)"
echo "Dataset: VHM_dataset_grpo_vg_only_2k (same 843 examples)"
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
    --model_name_or_path /arf/scratch/aalatan/Re-CoT/Qwen-VL-Series-Finetune/output/qwen35-cot-vg-merged \
    --dataset_name /arf/scratch/aalatan/VHM_dataset_grpo_vg_only_2k \
    --reward_funcs accuracy_shaped \
    --max_completion_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --bf16 true \
    --beta 0.003 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --sync_ref_model true \
    --ref_model_sync_steps 200 \
    --temperature 1.0 \
    --report_to wandb \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --min_pixels 200704 \
    --max_pixels 1003520 \
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
