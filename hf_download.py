from huggingface_hub import snapshot_download

# Download only the DOIR-RSVG subfolder from the dataset
snapshot_download(
    repo_id="aybora/VHM_dataset_sft",
    local_dir="/arf/scratch/aalatan/DOIR-RSVG",
    repo_type="dataset",
    allow_patterns="DOIR-RSVG/*",
    token=True  # This will use the token from huggingface-cli login
)