#!/usr/bin/env python3
"""
Create GRPO dataset for CLS with 300 misclassified (30%) + 700 correct (70%) samples.
This script:
1. Reads CLS evaluation results to find misclassified and correctly classified samples
2. Samples 300 misclassified and 700 correctly classified samples
3. Matches them with the original CoT dataset
4. FILTERS OUT images larger than 1000x1000 pixels
5. Creates a HuggingFace dataset in Arrow format matching the GRPO format
"""

import json
import base64
import random
import io
from pathlib import Path
from datasets import Dataset, DatasetDict, Image as ImageFeature
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

def check_image_size(image_path, max_size=1000):
    """
    Check if image dimensions are within max_size x max_size.
    Returns (is_valid, width, height).
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            is_valid = width <= max_size and height <= max_size
            return is_valid, width, height
    except Exception as e:
        print(f"  Error checking image size: {e}")
        return False, 0, 0

def load_eval_results(eval_json_path, target_misclassified=300, target_correct=700):
    """Load evaluation results and sample specified counts of each type."""
    with open(eval_json_path, 'r') as f:
        eval_data = json.load(f)

    # Separate by score
    misclassified = [item for item in eval_data if item['score'] == False]
    correctly_classified = [item for item in eval_data if item['score'] == True]

    print(f"Total samples: {len(eval_data)}")
    print(f"Misclassified samples available: {len(misclassified)}")
    print(f"Correctly classified samples available: {len(correctly_classified)}")

    # Sample target counts from each (or all if less than target)
    num_misclassified = min(target_misclassified, len(misclassified))
    num_correct = min(target_correct, len(correctly_classified))

    sampled_misclassified = random.sample(misclassified, num_misclassified)
    sampled_correct = random.sample(correctly_classified, num_correct)

    print(f"\nSampling:")
    print(f"  {num_misclassified} misclassified samples")
    print(f"  {num_correct} correctly classified samples")
    print(f"  Total: {num_misclassified + num_correct} samples")

    return sampled_misclassified, sampled_correct

def load_cot_dataset(cot_json_path):
    """Load the original CoT dataset with conversations."""
    with open(cot_json_path, 'r') as f:
        cot_data = json.load(f)

    # Create a mapping from image filename to full entry
    image_to_entry = {}
    for entry in cot_data:
        image_name = entry['image']
        image_to_entry[image_name] = entry

    print(f"Loaded {len(cot_data)} entries from CoT dataset")

    return image_to_entry

def load_image_as_pil(image_path):
    """Load image as PIL Image object."""
    return Image.open(image_path).convert('RGB')

def create_grpo_entry(item, cot_dataset, image_dir, max_image_size=1000):
    """Create a single GRPO dataset entry from an eval item. Filters out images > max_image_size."""
    filename = item['filename']

    # Check if this image exists in CoT dataset
    if filename not in cot_dataset:
        print(f"Warning: {filename} not found in CoT dataset")
        return None

    cot_entry = cot_dataset[filename]

    # Find the image file
    image_path = Path(image_dir) / filename
    if not image_path.exists():
        print(f"Warning: Image file not found: {image_path}")
        return None

    # Check image size - FILTER OUT if too large
    is_valid, width, height = check_image_size(image_path, max_size=max_image_size)
    if not is_valid:
        print(f"  Skipping {filename}: {width}x{height} exceeds {max_image_size}x{max_image_size}")
        return None

    # Load image as PIL Image (no resizing)
    pil_image = load_image_as_pil(image_path)

    # Extract question and solution from conversations
    conversations = cot_entry['conversations']

    # Assuming conversations[0] is human (question) and conversations[1] is gpt (answer/solution)
    question_value = conversations[0]['value']
    solution_value = conversations[1]['value']

    # Remove <image> tag from question if present
    question_text = question_value.replace('<image>\n', '').replace('<image>', '').strip()

    # Add instruction to question (matching GRPO format)
    question_with_instruction = f"{question_text} Make your chain of thought reasoning and then answer the question using a single word or phrase."

    # Extract answer from solution (text after <answer> tag)
    answer_text = ""
    if "<answer>" in solution_value and "</answer>" in solution_value:
        answer_text = solution_value.split("<answer>")[1].split("</answer>")[0].strip()

    # Encode original image to base64 for prompt
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Create solution as list: [turn_number, answer, question, image_base64]
    solution_list = [
        "1",  # Turn number (answer_type for classification)
        answer_text,  # Extracted answer
        question_with_instruction,  # Question with instruction
        image_base64  # Base64 encoded image
    ]

    # Create prompt in the GRPO format (list of message dicts with role and content)
    prompt_list = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning><answer> answer here </answer>"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "text": None
                },
                {
                    "type": "text",
                    "text": question_with_instruction
                }
            ]
        }
    ]

    # Create entry matching GRPO format
    entry = {
        'image': pil_image,
        'question': question_with_instruction,
        'solution': solution_list,
        'prompt': prompt_list
    }

    return entry

def create_balanced_grpo_dataset(misclassified, correctly_classified, cot_dataset, image_dir, output_dir, max_image_size=1000):
    """Create GRPO dataset with specified misclassified and correct samples."""

    # Combine all samples
    all_samples = misclassified + correctly_classified

    # Shuffle the combined samples
    random.shuffle(all_samples)

    print(f"\nProcessing {len(all_samples)} total samples...")

    dataset_entries = []
    skipped = 0
    misclassified_count = 0
    correct_count = 0

    for i, item in enumerate(all_samples, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(all_samples)} samples...")

        entry = create_grpo_entry(item, cot_dataset, image_dir, max_image_size=max_image_size)
        if entry is None:
            skipped += 1
            continue

        # Track counts for verification
        if item['score'] == False:
            misclassified_count += 1
        else:
            correct_count += 1

        dataset_entries.append(entry)

    print(f"\nSuccessfully processed {len(dataset_entries)} samples")
    print(f"  - Misclassified: {misclassified_count}")
    print(f"  - Correctly classified: {correct_count}")
    print(f"  - Skipped: {skipped}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(dataset_entries)

    # Cast the image column to Image feature type to match reference dataset
    dataset = dataset.cast_column('image', ImageFeature(decode=True))

    # Print the features to verify
    print("\nDataset features:")
    for col, feat in dataset.features.items():
        print(f"  {col}: {feat}")

    # Save as DatasetDict with 'train' split
    dataset_dict = DatasetDict({'train': dataset})

    # Save to disk in Arrow format
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples in dataset: {len(dataset)}")

    return dataset_dict

def main():
    # Paths for CLS dataset
    eval_json_path = "/arf/scratch/aalatan/FewShotReasoning/eval/eval_results/CLS_self-eval/CLS_self-eval_Qwen-2VL-2B-CLS-CoT_eval.json"
    cot_json_path = "/arf/scratch/aalatan/Datasets_Self-Eval_CLS/CLS_self-eval.json"
    image_dir = "/arf/scratch/aalatan/Datasets_Self-Eval_CLS/CLS_self-eval"
    output_dir = "/arf/scratch/aalatan/VHM_dataset_grpo_cls_300_700"

    # Configuration
    TARGET_MISCLASSIFIED = 300  # 30% hard (misclassified)
    TARGET_CORRECT = 700  # 70% easy (correctly classified)
    MAX_IMAGE_SIZE = 1000  # Max dimension in pixels

    print("="*60)
    print("CREATING CLS GRPO DATASET (70% EASY / 30% HARD)")
    print("="*60)
    print(f"Target: {TARGET_MISCLASSIFIED} misclassified (30%) + {TARGET_CORRECT} correct (70%)")
    print(f"Filtering: Only images <= {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} pixels")
    print("="*60)

    # Load data
    print("\nLoading evaluation results...")
    misclassified, correctly_classified = load_eval_results(eval_json_path, target_misclassified=TARGET_MISCLASSIFIED, target_correct=TARGET_CORRECT)

    print("\nLoading CoT dataset...")
    cot_dataset = load_cot_dataset(cot_json_path)

    # Create GRPO dataset
    print("\nCreating GRPO dataset...")
    dataset = create_balanced_grpo_dataset(
        misclassified,
        correctly_classified,
        cot_dataset,
        image_dir,
        output_dir,
        max_image_size=MAX_IMAGE_SIZE
    )

    print("\n" + "="*60)
    print("✓ CLS GRPO DATASET CREATION COMPLETED!")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Total samples: {len(dataset['train'])}")
    print(f"Distribution: 70% easy (correct) / 30% hard (misclassified)")
    print(f"All images are <= {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE} pixels")
    print("="*60)

if __name__ == "__main__":
    main()
