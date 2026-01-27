from huggingface_hub import HfApi

api = HfApi()

# Create the repository
api.create_repo(
    repo_id="yakupmrt7/Qwen-VL-2B-GRPO-VQA-363_EX_88epoch",
    repo_type="model",
    private=True  # or False for public
)

# Then upload
api.upload_folder(
    folder_path="/arf/scratch/aalatan/FewShotReasoning/train/checkpoints/Qwen-VL-2B-GRPO-VQA-363_EX_88epoch/final",
    repo_id="yakupmrt7/Qwen-VL-2B-GRPO-VQA-363_EX_88epoch",
    repo_type="model",
    commit_message="Upload full model"
)