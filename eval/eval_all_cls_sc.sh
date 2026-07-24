#!/bin/bash
# Submit all CLS evaluation tasks using Self-Consistency (majority voting).
# Generates N_SAMPLES responses per image at TEMPERATURE, then takes majority vote.
# Usage: ./eval_all_cls_sc.sh
# Only change MODEL_PATH and SC params below!

MODEL_PATH="/arf/scratch/aalatan/FewShotReasoning/train/checkpoints/Qwen-VL-2B-GRPO-500Easy-500Hard-Temp1.5-2epoch-cosine2026-02-24-14-42-14/final"

# Self-Consistency parameters
N_SAMPLES=7
SC_TEMPERATURE=0.6

# ============== DO NOT MODIFY BELOW ==============

# Extract model name from path (parent folder of "final")
MODEL_NAME=$(basename $(dirname $MODEL_PATH))

# All CLS tasks
CLS_TASKS=("cls_aid" "cls_METER_ML" "cls_NWPU_RESISC45" "cls_SIRI" "cls_WHU_RS19")

# Base directories
SC_SCRIPT_PATH="/arf/scratch/aalatan/FewShotReasoning/eval/python_script/evaluation/rs_evaluation_sc.py"
DATA_ROOT="/arf/scratch/aalatan/datasets_eval"
REASONING_CONFIG="/arf/scratch/aalatan/FewShotReasoning/eval/config/qwen2_thinking_template.json"
EVAL_RESULTS_BASE="/arf/scratch/aalatan/FewShotReasoning/eval/eval_results"

echo "=========================================="
echo "Submitting CLS Self-Consistency evaluation jobs"
echo "Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "n_samples: $N_SAMPLES  |  temperature: $SC_TEMPERATURE"
echo "=========================================="

for TASK in "${CLS_TASKS[@]}"; do
    OUTPUT_DIR="${EVAL_RESULTS_BASE}/${MODEL_NAME}/${TASK}_sc"

    # Create temporary job script
    JOB_SCRIPT=$(mktemp /tmp/eval_sc_${TASK}_XXXXXX.sh)

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=eval_sc_${TASK}
#SBATCH --output=eval_sc-%j.out
#SBATCH --error=eval_sc-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH -C H100
#SBATCH --exclude=kolyoz21

# Don't source bashrc - manually activate conda environment (scorers2)
export PATH="/arf/home/aalatan/mert/envs/scorers2/bin:\$PATH"
export CONDA_PREFIX="/arf/home/aalatan/mert/envs/scorers2"
export CONDA_DEFAULT_ENV="scorers2"
export LD_LIBRARY_PATH="/arf/home/aalatan/mert/envs/scorers2/lib:\$LD_LIBRARY_PATH"

cd /arf/scratch/aalatan/FewShotReasoning/eval

# Fix PYTHONPATH - include eval directory and conda packages
export PYTHONPATH=/arf/scratch/aalatan/FewShotReasoning/eval:/arf/home/aalatan/mert/envs/scorers2/lib/python3.10/site-packages:\$PYTHONPATH
export PYTHONUNBUFFERED=1
export DISABLE_FLASH_ATTN=1

mkdir -p ${OUTPUT_DIR}

echo "========== Job Info =========="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "GPU: 1"
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Output: ${OUTPUT_DIR}"
echo "Self-Consistency: n_samples=${N_SAMPLES}, temperature=${SC_TEMPERATURE}"
echo "PYTHONPATH: \$PYTHONPATH"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 ${SC_SCRIPT_PATH} \\
    --data_root ${DATA_ROOT} \\
    --output_dir ${OUTPUT_DIR} \\
    --model_type lmdeploy \\
    --model_path ${MODEL_PATH} \\
    --force_inference true \\
    --task ${TASK} \\
    --reasoning_config ${REASONING_CONFIG} \\
    --temperature ${SC_TEMPERATURE} \\
    --do_sample true \\
    --n_samples ${N_SAMPLES}

echo "Self-Consistency evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
EOF

    # Submit the job
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "Submitted ${TASK} (SC n=${N_SAMPLES}) -> Job ID: ${JOB_ID}"

    # Clean up temp script
    rm "$JOB_SCRIPT"
done

echo "=========================================="
echo "All 5 CLS self-consistency jobs submitted!"
echo "Results will be saved to: ${EVAL_RESULTS_BASE}/${MODEL_NAME}/<task>_sc/"
echo "=========================================="
