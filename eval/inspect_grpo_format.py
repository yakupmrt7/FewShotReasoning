#!/usr/bin/env python3
"""Quick script to inspect the GRPO dataset format."""

from datasets import load_from_disk

dataset_path = "/arf/scratch/aalatan/VHM_dataset_grpo_vqa_only_2k_proper"
dataset = load_from_disk(dataset_path)

print("Dataset structure:", dataset)
print("\nFirst sample:")
sample = dataset['train'][0]

for key, value in sample.items():
    if key == 'image':
        print(f"{key}: [base64, length={len(value)}]")
    elif isinstance(value, str) and len(value) > 200:
        print(f"{key}: {value[:200]}... [length={len(value)}]")
    else:
        print(f"{key}: {value}")

print("\n\nSecond sample:")
sample2 = dataset['train'][1]

for key, value in sample2.items():
    if key == 'image':
        print(f"{key}: [base64, length={len(value)}]")
    elif isinstance(value, str) and len(value) > 200:
        print(f"{key}: {value[:200]}... [length={len(value)}]")
    else:
        print(f"{key}: {value}")
