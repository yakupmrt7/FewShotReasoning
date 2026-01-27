from huggingface_hub import snapshot_download

# Download the private model
snapshot_download(
    repo_id="yakupmrtr/Qwen2VL-Insturct-CLS-64_64_2",
    local_dir="./Qwen2VL-Insturct-CLS-64_64_2",
    repo_type="model",
    token=True  # This will use the token from huggingface-cli login
)