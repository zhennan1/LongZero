#!/usr/bin/env python3
"""
Synthesize MRCR Dataset from Pre-generated Response Pool

This script builds MRCR dataset samples by randomly selecting responses from
a pre-generated pool. No API calls are made during synthesis.

Features:
- Uses Qwen tokenizer with chat template for accurate token counting
- Needles are randomly distributed in the conversation
- All responses are randomly selected from pool (no uniqueness guarantee)
- Fast synthesis without API calls

Usage:
    python synthesize_from_pool.py --pool_path response_pool.jsonl
    python synthesize_from_pool.py --length_config 8k-16k --needle_config 4  # Single config
    python synthesize_from_pool.py --length_config 8k-16k --needle_config 8  # Single config
"""

import os
import json
import random
import string
import argparse
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass

# Configuration
DEFAULT_POOL_PATH = "assets/mrcr_synth_v2/response_pool_100.jsonl"
DEFAULT_OUTPUT_DIR = "assets/mrcr_synth_v2"
DEFAULT_TOKENIZER_PATH = "assets/Qwen3-4B-Thinking-2507"

# Length and needle configurations
LENGTH_CONFIGS = [
    (4096, 8192, "4k-8k"),
    (8193, 16384, "8k-16k"),
    (16385, 32768, "16k-32k"),
]
NEEDLE_CONFIGS = [2, 4, 8]


def generate_random_string(length: int = 10) -> str:
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def get_ordinal(n: int) -> str:
    """Get ordinal string for a number."""
    if n == 1:
        return "1st"
    elif n == 2:
        return "2nd"
    elif n == 3:
        return "3rd"
    else:
        return f"{n}th"


def load_response_pool(path: str) -> Dict[Tuple[str, str], List[str]]:
    """Load response pool from JSONL file."""
    pool = defaultdict(list)

    print(f"Loading response pool from {path}...")
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                key = (item["format"], item["entity"])
                pool[key].append(item["response"])
            except:
                continue

    total = sum(len(v) for v in pool.values())
    print(f"Loaded {total} responses for {len(pool)} combinations")

    # Print distribution
    counts = [len(v) for v in pool.values()]
    print(f"Responses per combo: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")

    return pool


class MRCRSynthesizer:
    def __init__(
        self,
        pool: Dict[Tuple[str, str], List[str]],
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    ):
        self.pool = pool
        self.all_combos = list(pool.keys())

        # Load Qwen tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Extract unique formats and entities
        self.formats = list(set(k[0] for k in self.all_combos))
        self.entities = list(set(k[1] for k in self.all_combos))

        # Pre-generate example blocks
        self.example_blocks: List[str] = []
        self._pre_generate_example_blocks(100)

        print(f"Initialized synthesizer with {len(self.formats)} formats and {len(self.entities)} entities")

    def count_tokens_with_template(self, messages: List[Dict]) -> int:
        """Count tokens using chat template."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return len(self.tokenizer.encode(text))

    def get_random_response(self, key: Tuple[str, str], exclude: Set[str] = None) -> str:
        """Get a random response from pool, optionally excluding some."""
        responses = self.pool.get(key, [])
        if not responses:
            return None

        if exclude:
            available = [r for r in responses if r not in exclude]
            if available:
                return random.choice(available)

        return random.choice(responses)

    def _create_example_block(self) -> str:
        """Create an example block for the prompt prefix."""
        # Select two different combos
        combo1 = random.choice(self.all_combos)
        combo2 = random.choice([c for c in self.all_combos if c != combo1])

        format1, entity1 = combo1
        format2, entity2 = combo2

        response1_1 = self.get_random_response(combo1)
        response1_2 = self.get_random_response(combo1, exclude={response1_1} if response1_1 else None)
        response2 = self.get_random_response(combo2)

        if not all([response1_1, response1_2, response2]):
            # Fallback: use same response
            response1_2 = response1_1

        random_prefix = generate_random_string()

        example = f"""======EXAMPLE======
User: write a {format1} about {entity1}
Assistant: {response1_1}
User: write a {format2} about {entity2}
Assistant: {response2}
User: write a {format1} about {entity1}
Assistant: {response1_2}
User: Prepend {random_prefix} to the 1st (1 indexed) {format1} about {entity1}. Do not include any other text in your response.
Assistant: {random_prefix}{response1_1}

======END EXAMPLE======"""
        return example

    def _pre_generate_example_blocks(self, num_blocks: int = 100):
        """Pre-generate example blocks."""
        print(f"Pre-generating {num_blocks} example blocks...")
        for _ in range(num_blocks):
            block = self._create_example_block()
            self.example_blocks.append(block)

    def get_example_block(self) -> str:
        """Get a random example block."""
        return random.choice(self.example_blocks)

    def synthesize_sample(
        self,
        num_needles: int,
        min_tokens: int,
        max_tokens: int,
    ) -> Dict:
        """Synthesize a single MRCR-like sample."""
        # 1. Select target format + entity
        target_key = random.choice(self.all_combos)
        target_format, target_entity = target_key

        # 2. Get needle responses (randomly from pool, unique within sample)
        needle_responses = []
        used_in_sample = set()

        for _ in range(num_needles):
            resp = self.get_random_response(target_key, exclude=used_in_sample)
            if resp:
                needle_responses.append(resp)
                used_in_sample.add(resp)
            else:
                # Pool exhausted for this combo, allow duplicates
                resp = self.get_random_response(target_key)
                needle_responses.append(resp)

        if len(needle_responses) < num_needles:
            return None  # Not enough responses

        # 3. Determine target token count
        target_tokens = random.randint(min_tokens, max_tokens)

        # 4. Build example prefix
        example_prefix = "Here are some examples of conversations succeeded by a follow-up question answered correctly:\n\n"
        example_prefix += self.get_example_block()
        example_prefix += "\n\n" + self.get_example_block()

        # 5. Initialize messages
        messages = [{"role": "user", "content": example_prefix}]

        # Get distractor combinations (exclude target)
        distractor_combos = [k for k in self.all_combos if k != target_key]

        # 6. Build distractor turns
        # Reserve space for needles (estimate ~350 tokens each)
        distractor_target = target_tokens - num_needles * 350
        distractor_turns = []

        current_tokens = self.count_tokens_with_template(messages)

        while current_tokens < distractor_target:
            dist_key = random.choice(distractor_combos)
            dist_format, dist_entity = dist_key
            user_msg = f"write a {dist_format} about {dist_entity}"

            assistant_msg = self.get_random_response(dist_key, exclude=used_in_sample)
            if assistant_msg is None:
                assistant_msg = self.get_random_response(dist_key)

            if assistant_msg is None:
                continue

            used_in_sample.add(assistant_msg)
            distractor_turns.append((user_msg, assistant_msg))

            # Update token count
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
            current_tokens = self.count_tokens_with_template(messages)

            # Safety check
            if len(distractor_turns) > 500:
                break

        # Reset messages for final assembly
        messages = [{"role": "user", "content": example_prefix}]

        # 7. Randomly select positions for needles
        total_positions = len(distractor_turns) + 1

        # Ensure we have enough positions
        if total_positions < num_needles:
            # Add more distractor turns
            while total_positions < num_needles + 1:
                dist_key = random.choice(distractor_combos)
                user_msg = f"write a {dist_key[0]} about {dist_key[1]}"
                assistant_msg = self.get_random_response(dist_key)
                if assistant_msg:
                    distractor_turns.append((user_msg, assistant_msg))
                    total_positions += 1

        # Random needle positions (sorted)
        needle_positions = sorted(random.sample(range(total_positions), num_needles))

        # 8. Assemble final conversation
        needle_idx = 0
        distractor_idx = 0

        for pos in range(total_positions):
            if needle_idx < num_needles and pos == needle_positions[needle_idx]:
                # Insert needle
                messages.append({"role": "user", "content": f"write a {target_format} about {target_entity}"})
                messages.append({"role": "assistant", "content": needle_responses[needle_idx]})
                needle_idx += 1
            else:
                # Insert distractor
                if distractor_idx < len(distractor_turns):
                    user_msg, assistant_msg = distractor_turns[distractor_idx]
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                    distractor_idx += 1

        # 9. Add final question
        desired_needle_index = random.randint(1, num_needles)
        random_string = generate_random_string()
        ordinal = get_ordinal(desired_needle_index)

        final_question = (
            f"Prepend {random_string} to the {ordinal} (1 indexed) "
            f"{target_format} about {target_entity}. "
            f"Do not include any other text in your response."
        )
        messages.append({"role": "user", "content": final_question})

        # 10. Calculate answer and final token count
        answer = random_string + needle_responses[desired_needle_index - 1]
        final_tokens = self.count_tokens_with_template(messages)

        # 11. Adjust if out of range
        if final_tokens > max_tokens:
            # Remove some distractor turns
            removed = 0
            while final_tokens > max_tokens and len(messages) > 5:
                # Find and remove a distractor turn (not needle, not first, not last)
                for i in range(len(messages) - 3, 2, -2):
                    if (messages[i]["role"] == "user" and
                        messages[i]["content"] != f"write a {target_format} about {target_entity}"):
                        messages.pop(i + 1)  # Remove assistant
                        messages.pop(i)      # Remove user
                        removed += 1
                        break
                else:
                    break  # No more distractors to remove
                final_tokens = self.count_tokens_with_template(messages)

        # Verify token count is in range
        if final_tokens < min_tokens or final_tokens > max_tokens:
            # Out of range, return None to retry
            return None

        return {
            "prompt": json.dumps(messages, ensure_ascii=False),
            "answer": answer,
            "random_string_to_prepend": random_string,
            "n_needles": num_needles,
            "desired_needle_index": desired_needle_index,
            "total_messages": len(messages),
            "n_tokens": final_tokens,
            "target_format": target_format,
            "target_entity": target_entity,
            "needle_positions": needle_positions,
        }

    def synthesize_dataset(
        self,
        num_samples: int,
        num_needles: int,
        min_tokens: int,
        max_tokens: int,
        output_path: str,
        save_interval: int = 100,
    ) -> List[Dict]:
        """Synthesize a dataset with real-time saving."""
        samples = []

        # Load existing samples if file exists (for resume)
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except:
                        pass
            print(f"Loaded {len(samples)} existing samples from {output_path}")

        start_idx = len(samples)
        if start_idx >= num_samples:
            print(f"Already have {start_idx} samples, skipping...")
            return samples

        print(f"Synthesizing {num_samples - start_idx} samples (starting from {start_idx})...")
        print(f"Config: {num_needles} needles, {min_tokens}-{max_tokens} tokens")

        pbar = tqdm(total=num_samples, initial=start_idx, desc="Generating samples")

        retries = 0
        max_retries = 10

        while len(samples) < num_samples:
            sample = self.synthesize_sample(
                num_needles=num_needles,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )

            if sample:
                samples.append(sample)
                retries = 0

                # Real-time save
                if len(samples) % save_interval == 0:
                    with open(output_path, 'w') as f:
                        for s in samples:
                            f.write(json.dumps(s, ensure_ascii=False) + '\n')

                pbar.update(1)
            else:
                retries += 1
                if retries > max_retries:
                    print(f"Warning: Failed to generate sample after {max_retries} retries")
                    retries = 0

        pbar.close()

        # Final save
        with open(output_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

        return samples


def main():
    parser = argparse.ArgumentParser(description="Synthesize MRCR dataset from response pool")
    parser.add_argument("--pool_path", type=str, default=DEFAULT_POOL_PATH, help="Path to response pool JSONL")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer")
    parser.add_argument("--num_train", type=int, default=20480, help="Number of training samples per config")
    parser.add_argument("--num_val", type=int, default=512, help="Number of validation samples per config")
    parser.add_argument("--length_config", type=str, default=None, help="Specific length config (e.g., '8k-16k')")
    parser.add_argument("--needle_config", type=int, default=None, help="Specific needle config (e.g., 4)")

    args = parser.parse_args()

    # Load response pool
    pool = load_response_pool(args.pool_path)

    # Initialize synthesizer
    synthesizer = MRCRSynthesizer(
        pool=pool,
        tokenizer_path=args.tokenizer_path,
    )

    # Determine which configs to run
    length_configs = LENGTH_CONFIGS
    needle_configs = NEEDLE_CONFIGS

    if args.length_config:
        length_configs = [c for c in LENGTH_CONFIGS if c[2] == args.length_config]
        if not length_configs:
            print(f"Invalid length config: {args.length_config}")
            print(f"Available: {[c[2] for c in LENGTH_CONFIGS]}")
            return

    if args.needle_config:
        if args.needle_config not in NEEDLE_CONFIGS:
            print(f"Invalid needle config: {args.needle_config}")
            print(f"Available: {NEEDLE_CONFIGS}")
            return
        needle_configs = [args.needle_config]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate all configurations
    total_samples = 0
    for min_tok, max_tok, length_name in length_configs:
        for num_needles in needle_configs:
            config_name = f"{length_name}_{num_needles}needle_2"
            output_subdir = os.path.join(args.output_dir, config_name)
            os.makedirs(output_subdir, exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"Generating: {config_name}")
            print(f"{'=' * 60}")

            # Generate training set
            train_path = os.path.join(output_subdir, "train.jsonl")
            print(f"\n--- Training Set ---")
            train_samples = synthesizer.synthesize_dataset(
                num_samples=args.num_train,
                num_needles=num_needles,
                min_tokens=min_tok,
                max_tokens=max_tok,
                output_path=train_path,
            )
            total_samples += len(train_samples)

            # Generate validation set
            val_path = os.path.join(output_subdir, "val.jsonl")
            print(f"\n--- Validation Set ---")
            val_samples = synthesizer.synthesize_dataset(
                num_samples=args.num_val,
                num_needles=num_needles,
                min_tokens=min_tok,
                max_tokens=max_tok,
                output_path=val_path,
            )
            total_samples += len(val_samples)

            # Print stats for this config
            if train_samples:
                tokens = [s["n_tokens"] for s in train_samples]
                print(f"Train token range: {min(tokens)} - {max(tokens)}")

    print(f"\n{'=' * 60}")
    print("SYNTHESIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total samples generated: {total_samples}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
