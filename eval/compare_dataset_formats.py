#!/usr/bin/env python3
"""
Compare two HuggingFace datasets to ensure they have matching formats.
This script checks:
1. Dataset structure (splits, number of samples)
2. Schema (column names, data types)
3. Entry format (field structure, data format)
4. Image encoding format (base64)
"""

import json
from pathlib import Path
from datasets import load_from_disk
import base64

def check_base64_validity(b64_string, sample_name=""):
    """Check if a string is valid base64."""
    try:
        decoded = base64.b64decode(b64_string)
        return True, len(decoded)
    except Exception as e:
        return False, str(e)

def compare_datasets(dataset1_path, dataset2_path):
    """Compare two datasets and report differences."""

    print("="*80)
    print("DATASET FORMAT COMPARISON")
    print("="*80)

    # Load datasets
    print(f"\nLoading dataset 1: {dataset1_path}")
    dataset1 = load_from_disk(dataset1_path)

    print(f"Loading dataset 2: {dataset2_path}")
    dataset2 = load_from_disk(dataset2_path)

    print("\n" + "="*80)
    print("1. DATASET STRUCTURE")
    print("="*80)

    # Check splits
    splits1 = list(dataset1.keys())
    splits2 = list(dataset2.keys())

    print(f"\nDataset 1 splits: {splits1}")
    print(f"Dataset 2 splits: {splits2}")

    if splits1 == splits2:
        print("✓ Splits match!")
    else:
        print("✗ Splits DO NOT match!")
        return False

    # Check number of samples in each split
    for split in splits1:
        len1 = len(dataset1[split])
        len2 = len(dataset2[split])
        print(f"\n'{split}' split:")
        print(f"  Dataset 1: {len1} samples")
        print(f"  Dataset 2: {len2} samples")

    print("\n" + "="*80)
    print("2. SCHEMA COMPARISON")
    print("="*80)

    all_match = True

    for split in splits1:
        print(f"\nSplit: '{split}'")

        # Get column names
        cols1 = dataset1[split].column_names
        cols2 = dataset2[split].column_names

        print(f"  Dataset 1 columns: {cols1}")
        print(f"  Dataset 2 columns: {cols2}")

        if cols1 == cols2:
            print("  ✓ Column names match!")
        else:
            print("  ✗ Column names DO NOT match!")
            all_match = False
            continue

        # Get features (data types)
        features1 = dataset1[split].features
        features2 = dataset2[split].features

        print(f"\n  Dataset 1 features:")
        for col, feat in features1.items():
            print(f"    {col}: {feat}")

        print(f"\n  Dataset 2 features:")
        for col, feat in features2.items():
            print(f"    {col}: {feat}")

        if str(features1) == str(features2):
            print("\n  ✓ Features (data types) match!")
        else:
            print("\n  ✗ Features (data types) DO NOT match!")
            all_match = False

    print("\n" + "="*80)
    print("3. SAMPLE ENTRY COMPARISON")
    print("="*80)

    for split in splits1:
        if len(dataset1[split]) > 0 and len(dataset2[split]) > 0:
            print(f"\nComparing first entry in '{split}' split:")

            sample1 = dataset1[split][0]
            sample2 = dataset2[split][0]

            # Check keys
            keys1 = set(sample1.keys())
            keys2 = set(sample2.keys())

            print(f"\n  Dataset 1 keys: {sorted(keys1)}")
            print(f"  Dataset 2 keys: {sorted(keys2)}")

            if keys1 == keys2:
                print("  ✓ Entry keys match!")
            else:
                print("  ✗ Entry keys DO NOT match!")
                print(f"    Missing in dataset 1: {keys2 - keys1}")
                print(f"    Missing in dataset 2: {keys1 - keys2}")
                all_match = False
                continue

            # Check each field
            for key in sorted(keys1):
                val1 = sample1[key]
                val2 = sample2[key]

                print(f"\n  Field: '{key}'")
                print(f"    Dataset 1 type: {type(val1).__name__}")
                print(f"    Dataset 2 type: {type(val2).__name__}")

                if type(val1) != type(val2):
                    print(f"    ✗ Types DO NOT match!")
                    all_match = False
                    continue

                # Special handling for different types
                if key == 'image':
                    # Check if it's base64 encoded
                    is_valid1, info1 = check_base64_validity(val1, "dataset1")
                    is_valid2, info2 = check_base64_validity(val2, "dataset2")

                    print(f"    Dataset 1 base64: {'Valid' if is_valid1 else 'Invalid'} ({info1} bytes)" if is_valid1 else f"    Dataset 1 base64: Invalid - {info1}")
                    print(f"    Dataset 2 base64: {'Valid' if is_valid2 else 'Invalid'} ({info2} bytes)" if is_valid2 else f"    Dataset 2 base64: Invalid - {info2}")

                    if is_valid1 and is_valid2:
                        print(f"    ✓ Both are valid base64 encoded images!")
                    else:
                        print(f"    ✗ One or both are NOT valid base64!")
                        all_match = False

                elif key == 'conversations':
                    # Check conversations structure
                    if isinstance(val1, list) and isinstance(val2, list):
                        print(f"    Dataset 1: {len(val1)} conversation turns")
                        print(f"    Dataset 2: {len(val2)} conversation turns")

                        if len(val1) > 0 and len(val2) > 0:
                            conv1_keys = set(val1[0].keys()) if isinstance(val1[0], dict) else set()
                            conv2_keys = set(val2[0].keys()) if isinstance(val2[0], dict) else set()

                            print(f"    Dataset 1 conv keys: {sorted(conv1_keys)}")
                            print(f"    Dataset 2 conv keys: {sorted(conv2_keys)}")

                            if conv1_keys == conv2_keys:
                                print(f"    ✓ Conversation structure matches!")
                            else:
                                print(f"    ✗ Conversation structure DOES NOT match!")
                                all_match = False
                    else:
                        print(f"    ✗ Conversations are not lists!")
                        all_match = False

                else:
                    # Generic comparison
                    if isinstance(val1, str):
                        print(f"    Dataset 1 length: {len(val1)} chars")
                        print(f"    Dataset 2 length: {len(val2)} chars")
                    print(f"    ✓ Type matches!")

    print("\n" + "="*80)
    print("4. DETAILED SAMPLE INSPECTION")
    print("="*80)

    for split in splits1:
        if len(dataset1[split]) > 0 and len(dataset2[split]) > 0:
            print(f"\nFirst sample from Dataset 1 ('{split}'):")
            sample1 = dataset1[split][0]

            for key, value in sample1.items():
                if key == 'image':
                    is_valid, info = check_base64_validity(value)
                    if is_valid:
                        print(f"  {key}: [base64 string, {info} bytes when decoded]")
                        print(f"         First 50 chars: {value[:50]}...")
                    else:
                        print(f"  {key}: [Invalid base64: {info}]")
                elif key == 'conversations':
                    print(f"  {key}:")
                    for i, conv in enumerate(value):
                        print(f"    [{i}] {conv}")
                else:
                    # Truncate long strings
                    if isinstance(value, str) and len(value) > 500:
                        print(f"  {key}: {value[:500]}... [truncated, total length: {len(value)}]")
                    else:
                        print(f"  {key}: {value}")

            print(f"\nFirst sample from Dataset 2 ('{split}'):")
            sample2 = dataset2[split][0]

            for key, value in sample2.items():
                if key == 'image':
                    is_valid, info = check_base64_validity(value)
                    if is_valid:
                        print(f"  {key}: [base64 string, {info} bytes when decoded]")
                        print(f"         First 50 chars: {value[:50]}...")
                    else:
                        print(f"  {key}: [Invalid base64: {info}]")
                elif key == 'conversations':
                    print(f"  {key}:")
                    for i, conv in enumerate(value):
                        # Truncate conversation values if too long
                        if isinstance(conv, dict):
                            truncated_conv = {}
                            for k, v in conv.items():
                                if isinstance(v, str) and len(v) > 500:
                                    truncated_conv[k] = v[:500] + f"... [truncated, total length: {len(v)}]"
                                else:
                                    truncated_conv[k] = v
                            print(f"    [{i}] {truncated_conv}")
                        else:
                            print(f"    [{i}] {conv}")
                else:
                    # Truncate long strings
                    if isinstance(value, str) and len(value) > 500:
                        print(f"  {key}: {value[:500]}... [truncated, total length: {len(value)}]")
                    else:
                        print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if all_match:
        print("\n✓✓✓ ALL CHECKS PASSED! Datasets have matching formats! ✓✓✓\n")
        return True
    else:
        print("\n✗✗✗ SOME CHECKS FAILED! Datasets have format differences! ✗✗✗\n")
        return False

def main():
    # Paths to compare
    dataset1_path = "/arf/scratch/aalatan/grpo_data/VHM_dataset_grpo_vqa_only_2k_proper"
    dataset2_path = "/arf/scratch/aalatan/grpo_data/VHM_dataset_grpo_vqa_self_eval_misclassified"

    print(f"\nDataset 1 (Reference): {dataset1_path}")
    print(f"Dataset 2 (New):       {dataset2_path}\n")

    result = compare_datasets(dataset1_path, dataset2_path)

    return 0 if result else 1

if __name__ == "__main__":
    exit(main())
