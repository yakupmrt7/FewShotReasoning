#!/bin/bash
#SBATCH --account=ogam6
#SBATCH --job-name=eval_vqa_363
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#SBATCH --partition=kolyoz-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH -C H100

# Don't source bashrc - manually activate conda environment (scorers2)
export PATH="/arf/home/aalatan/mert/envs/scorers2/bin:$PATH"
export CONDA_PREFIX="/arf/home/aalatan/mert/envs/scorers2"
export CONDA_DEFAULT_ENV="scorers2"
export LD_LIBRARY_PATH="/arf/home/aalatan/mert/envs/scorers2/lib:$LD_LIBRARY_PATH"

cd /arf/scratch/aalatan/FewShotReasoning/eval

# Fix PYTHONPATH - include eval directory and conda packages
export PYTHONPATH=/arf/scratch/aalatan/FewShotReasoning/eval:/arf/home/aalatan/mert/envs/scorers2/lib/python3.10/site-packages:$PYTHONPATH
export PYTHONUNBUFFERED=1
export DISABLE_FLASH_ATTN=1

SCRIPT_PATH="/arf/scratch/aalatan/FewShotReasoning/eval/python_script/evaluation/rs_evaluation.py"
DATA_ROOT="/arf/scratch/aalatan/Datasets_Self-Eval_CLS"
OUTPUT_DIR="/arf/scratch/aalatan/FewShotReasoning/eval/eval_results/CLS_self-eval"
model_type=lmdeploy
MODEL_PATH="/arf/scratch/aalatan/Qwen-2VL-2B-CLS-CoT"
REASONING_CONFIG="/arf/scratch/aalatan/FewShotReasoning/eval/config/qwen2_thinking_template.json"

mkdir -p $OUTPUT_DIR

echo "========== Job Info =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: 1"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 $SCRIPT_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --model_type $model_type \
    --model_path $MODEL_PATH \
    --force_inference true \
    --task CLS_self-eval \
    --reasoning_config $REASONING_CONFIG \

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"

#    "cls_aid": ("cls_AID.json", "cls"),
#    "cls_METER_ML": ("cls_METER_ML.json", "cls"),
#    "cls_NWPU_RESISC45": ("cls_NWPU_RESISC45.json", "cls"),
#    "cls_SIRI": ("cls_SIRI_WHU.json", "cls"),
#    "cls_WHU_RS19": ("cls_WHU_RS19.json", "cls"),



#    "vqa_HR-comp": ("RSVQA_HR-comp_RSVQA.json", "vqa"),
#    "vqa_HR-pre": ("RSVQA_HR-presence_RSVQA.json", "vqa"),
#    "vqa_LR-comp": ("RSVQA_LR-comp_RSVQA.json", "vqa"),
#    "vqa_LR-pre": ("RSVQA_LR-presence_RSVQA.json", "vqa"),
#    "vqa_LR-rural": ("RSVQA_LR-rural_urban_RSVQA.json", "vqa"),