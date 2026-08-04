#!/usr/bin/env python3
"""
Create BALANCED GRPO dataset with 1:1 ratio of misclassified to correctly classified samples.
This script:
1. Reads evaluation results to find misclassified and correctly classified samples
2. Samples equal number of correctly classified samples as misclassified ones
3. Matches them with the original CoT dataset
4. Creates a HuggingFace dataset in Arrow format matching the GRPO format
"""

import json
import base64
import random
from pathlib import Path
from datasets import Dataset, DatasetDict, Image as ImageFeature
from PIL import Image

# Set random seed for reproducibility
random.seed(42)

def load_eval_results(eval_json_path):
    """Load evaluation results and separate misclassified and correctly classified samples."""
    with open(eval_json_path, 'r') as f:
        eval_data = json.load(f)

    # Separate by score
    misclassified = [item for item in eval_data if item['score'] == False]
    correctly_classified = [item for item in eval_data if item['score'] == True]

    print(f"Total samples: {len(eval_data)}")
    print(f"Misclassified samples: {len(misclassified)}")
    print(f"Correctly classified samples: {len(correctly_classified)}")

    return misclassified, correctly_classified

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

def create_grpo_entry(item, cot_dataset, image_dir):
    """Create a single GRPO dataset entry from an eval item."""
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

    # Load image as PIL Image
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

    # Encode image to base64 for prompt
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Create solution as list: [turn_number, answer, question, image_base64]
    solution_list = [
        "1",  # Turn number
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

def create_balanced_grpo_dataset(misclassified, correctly_classified, cot_dataset, image_dir, output_dir):
    """Create balanced GRPO dataset with 1:1 ratio."""

    # Sample equal number of correctly classified as misclassified
    num_misclassified = len(misclassified)
    num_to_sample = min(num_misclassified, len(correctly_classified))

    print(f"\nBalanced sampling:")
    print(f"  Misclassified samples: {num_misclassified}")
    print(f"  Sampling {num_to_sample} from correctly classified (out of {len(correctly_classified)})")

    # Random sample from correctly classified
    sampled_correct = random.sample(correctly_classified, num_to_sample)

    # Combine all samples
    all_samples = misclassified + sampled_correct

    # Shuffle the combined samples
    random.shuffle(all_samples)

    print(f"  Total balanced samples: {len(all_samples)}")

    dataset_entries = []
    skipped = 0
    misclassified_count = 0
    correct_count = 0

    for item in all_samples:
        entry = create_grpo_entry(item, cot_dataset, image_dir)
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
    print(f"Skipped {skipped} samples")

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
    # Paths
    eval_json_path = "/arf/scratch/aalatan/FewShotReasoning/eval/eval_results/Self_eval/vqa_LR-pre/VQA_self-eval_Qwen2VL-2B-VQA-CoT_eval.json"
    cot_json_path = "/arf/scratch/aalatan/Datasets_Self-Eval_VQA/VQA_self-eval.json"
    image_dir = "/arf/scratch/aalatan/Datasets_Self-Eval_VQA/VQA_self-eval"
    output_dir = "/arf/scratch/aalatan/grpo_data/VHM_dataset_grpo_vqa_self_eval_balanced_1to1"

    # Load data
    print("Loading evaluation results...")
    misclassified, correctly_classified = load_eval_results(eval_json_path)

    print("\nLoading CoT dataset...")
    cot_dataset = load_cot_dataset(cot_json_path)

    # Create balanced GRPO dataset
    print("\nCreating balanced GRPO dataset (1:1 ratio)...")
    dataset = create_balanced_grpo_dataset(
        misclassified,
        correctly_classified,
        cot_dataset,
        image_dir,
        output_dir
    )

    print("\n" + "="*60)
    print("BALANCED GRPO DATASET CREATION COMPLETED!")
    print("="*60)
    print(f"Output: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
