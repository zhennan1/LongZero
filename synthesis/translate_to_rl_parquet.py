#!/usr/bin/env python3
"""
Convert MRCR JSONL dataset to verl RL parquet format.

This script converts the synthesized MRCR-like JSONL dataset to the parquet format
required by verl for GRPO training.

Usage:
    python translate_to_rl_parquet.py --jsonl_file train.jsonl --output_dir rl_data
"""

import os
import json
import argparse
import datasets
from pathlib import Path


# MRCR-specific prompt format (no thinking instructions, just the conversation)
# The prompt is already in the correct format from synthesis


def read_jsonl(file_path: str) -> list:
    """Read JSONL file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def translate_to_verl_format(jsonl_list: list, data_source_name: str, split: str = 'train') -> list:
    """
    Convert MRCR data to verl training format.

    Args:
        jsonl_list: List of MRCR samples
        data_source_name: Data source identifier
        split: 'train' or 'test'

    Returns:
        List of verl-formatted samples
    """
    new_list = []
    for idx, item in enumerate(jsonl_list):
        # Parse the prompt (it's stored as JSON string)
        prompt_messages = json.loads(item['prompt'])

        # Build verl format item
        new_item = {
            "data_source": f"custom_mrcr_{data_source_name}",
            "prompt": prompt_messages,  # Already in message format
            "ability": "mrcr",
            "reward_model": {
                "style": "rule",
                "ground_truth": item['answer'],
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'random_string_to_prepend': item['random_string_to_prepend'],
                'n_needles': item['n_needles'],
                'desired_needle_index': item.get('desired_needle_index', item.get('desired_msg_index')),
                'total_messages': item['total_messages'],
                'n_tokens': item.get('n_tokens', 0),
                'target_format': item.get('target_format', ''),
                'target_entity': item.get('target_entity', ''),
                'needle_positions': item.get('needle_positions', []),
                'answer': item['answer'],
            }
        }
        new_list.append(new_item)
    return new_list


def main():
    parser = argparse.ArgumentParser(description="Convert MRCR JSONL to verl parquet format")
    parser.add_argument(
        '--train_jsonl',
        type=str,
        default='assets/mrcr_synth_v2/8k-16k_8needle_2/train.jsonl',
        help='Path to training JSONL file'
    )
    parser.add_argument(
        '--val_jsonl',
        type=str,
        default='assets/mrcr_synth_v2/8k-16k_8needle_2/val.jsonl',
        help='Path to validation JSONL file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='assets/mrcr_synth_v2/8k-16k_8needle_2',
        help='Output directory for parquet files'
    )
    parser.add_argument(
        '--data_source_name',
        type=str,
        default='custom_mrcr_8needle_16k',
        help='Data source name identifier'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process training data
    print(f"Processing training data from {args.train_jsonl}...")
    train_data = read_jsonl(args.train_jsonl)
    print(f"  Loaded {len(train_data)} training samples")

    train_converted = translate_to_verl_format(train_data, args.data_source_name, split='train')
    print(f"  Converted {len(train_converted)} training samples")

    # Process validation data
    print(f"Processing validation data from {args.val_jsonl}...")
    val_data = read_jsonl(args.val_jsonl)
    print(f"  Loaded {len(val_data)} validation samples")

    val_converted = translate_to_verl_format(val_data, args.data_source_name, split='valid')
    print(f"  Converted {len(val_converted)} validation samples")

    # Convert to HuggingFace datasets
    train_dataset = datasets.Dataset.from_list(train_converted)
    val_dataset = datasets.Dataset.from_list(val_converted)

    # Save as parquet
    train_path = os.path.join(args.output_dir, 'train.parquet')
    val_path = os.path.join(args.output_dir, 'valid.parquet')

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"\nSaved files:")
    print(f"  Train: {train_path} ({len(train_dataset)} samples)")
    print(f"  Valid: {val_path} ({len(val_dataset)} samples)")

    # Print sample info
    print("\nSample data structure:")
    sample = train_converted[0]
    print(f"  data_source: {sample['data_source']}")
    print(f"  prompt: {len(sample['prompt'])} messages")
    print(f"  ability: {sample['ability']}")
    print(f"  reward_model.style: {sample['reward_model']['style']}")
    print(f"  extra_info keys: {list(sample['extra_info'].keys())}")

    print("\nConversion completed successfully!")


if __name__ == "__main__":
    main()
