"""
Dataset preprocessing for TTS evaluation.

Extracts navigation instructions from Touchdown and R2R datasets,
filters for appropriate length, removes duplicates, and samples test corpus.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


class NavigationInstructionExtractor:
    """Extracts and filters navigation instructions from datasets."""

    def __init__(self, data_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize extractor.

        Args:
            data_dir: Directory containing raw downloaded data
            output_dir: Directory to save processed samples
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "raw"
        if output_dir is None:
            output_dir = Path(__file__).parent

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw instruction text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?\'-]', '', text)

        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]

        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def extract_touchdown_instructions(self) -> List[Dict]:
        """
        Extract instructions from Touchdown dataset (JSONL format).

        Returns:
            List of instruction dictionaries
        """
        instructions = []

        touchdown_files = [
            self.data_dir / "touchdown" / "train.json",
            self.data_dir / "touchdown" / "dev.json"
        ]

        for file_path in touchdown_files:
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue

            # Touchdown is JSONL format (one JSON object per line)
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)

                        # Extract navigation_text field
                        if 'navigation_text' in item and item['navigation_text']:
                            nav_text = item['navigation_text'].strip()
                            if nav_text:
                                instructions.append({
                                    'source': 'touchdown',
                                    'source_file': file_path.name,
                                    'raw_text': nav_text
                                })

                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num + 1} in {file_path.name}: {e}")
                        continue

        return instructions

    def extract_r2r_instructions(self) -> List[Dict]:
        """
        Extract instructions from R2R dataset.

        Returns:
            List of instruction dictionaries
        """
        instructions = []

        r2r_files = [
            self.data_dir / "r2r" / "R2R_train.json",
            self.data_dir / "r2r" / "R2R_val_seen.json"
        ]

        for file_path in r2r_files:
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue

            with open(file_path, 'r') as f:
                data = json.load(f)

            # R2R format: list of navigation episodes with multiple instruction variants
            for idx, item in enumerate(data):
                if isinstance(item, dict) and 'instructions' in item:
                    # R2R has multiple paraphrases of the same route
                    for instruction in item['instructions']:
                        if isinstance(instruction, str):
                            instructions.append({
                                'source': 'r2r',
                                'source_file': file_path.name,
                                'raw_text': instruction
                            })

        return instructions

    def filter_instructions(self, instructions: List[Dict],
                           min_words: int = 5,
                           max_words: int = 50) -> List[Dict]:
        """
        Filter instructions by length and quality.

        Args:
            instructions: List of instruction dictionaries
            min_words: Minimum word count
            max_words: Maximum word count

        Returns:
            Filtered instructions
        """
        filtered = []

        for inst in instructions:
            raw_text = inst['raw_text']
            word_count = len(raw_text.split())

            # Filter by length
            if word_count < min_words or word_count > max_words:
                continue

            # Clean text
            cleaned = self.clean_text(raw_text)

            # Skip if cleaning removed too much
            if len(cleaned) < 10:
                continue

            # Skip if mostly numbers or special chars
            alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
            if alpha_ratio < 0.5:
                continue

            inst['text'] = cleaned
            inst['word_count'] = len(cleaned.split())
            filtered.append(inst)

        return filtered

    def remove_duplicates(self, instructions: List[Dict]) -> List[Dict]:
        """
        Remove duplicate instructions (case-insensitive).

        Args:
            instructions: List of instruction dictionaries

        Returns:
            Deduplicated instructions
        """
        seen = set()
        unique = []

        for inst in instructions:
            text_lower = inst['text'].lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique.append(inst)

        return unique

    def balance_sources(self, instructions: List[Dict],
                       target_count: int = 100) -> List[Dict]:
        """
        Sample instructions with balanced representation from both datasets.

        Args:
            instructions: List of instruction dictionaries
            target_count: Number of samples to select

        Returns:
            Balanced sample of instructions
        """
        # Group by source
        by_source = {}
        for inst in instructions:
            source = inst['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(inst)

        # Calculate samples per source (50/50 split if both available)
        sources = list(by_source.keys())
        samples_per_source = target_count // len(sources)

        sampled = []
        for source in sources:
            available = by_source[source]
            n_samples = min(samples_per_source, len(available))
            sampled.extend(random.sample(available, n_samples))

        # If we need more to reach target, sample from remaining
        if len(sampled) < target_count:
            remaining = target_count - len(sampled)
            all_remaining = [inst for inst in instructions if inst not in sampled]
            if all_remaining:
                additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
                sampled.extend(additional)

        # Shuffle final sample
        random.shuffle(sampled)

        return sampled[:target_count]

    def create_final_samples(self, instructions: List[Dict]) -> List[Dict]:
        """
        Create final sample list with IDs and metadata.

        Args:
            instructions: Filtered and sampled instructions

        Returns:
            Final sample list with IDs
        """
        samples = []

        for idx, inst in enumerate(instructions):
            sample = {
                'id': f"{inst['source']}_{idx:03d}",
                'text': inst['text'],
                'source': inst['source'],
                'word_count': inst['word_count']
            }
            samples.append(sample)

        return samples

    def save_samples(self, samples: List[Dict], output_path: Optional[Path] = None):
        """
        Save samples to JSON file.

        Args:
            samples: List of sample dictionaries
            output_path: Path to save samples (defaults to data/samples.json)
        """
        if output_path is None:
            output_path = self.output_dir / "samples.json"

        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"✓ Saved {len(samples)} samples to {output_path}")

    def print_statistics(self, instructions: List[Dict]):
        """
        Print statistics about extracted instructions.

        Args:
            instructions: List of instruction dictionaries
        """
        print("\n=== Dataset Statistics ===")

        # Count by source
        source_counts = Counter(inst['source'] for inst in instructions)
        print("\nSamples by source:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")

        # Word count statistics
        word_counts = [inst['word_count'] for inst in instructions]
        print(f"\nWord count statistics:")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")
        print(f"  Mean: {sum(word_counts) / len(word_counts):.1f}")

        # Show a few examples
        print("\nExample instructions:")
        for inst in random.sample(instructions, min(5, len(instructions))):
            print(f"  [{inst['source']}] {inst['text']}")


def preprocess_datasets(data_dir: Optional[Path] = None,
                       output_dir: Optional[Path] = None,
                       n_samples: int = 100,
                       seed: int = 42) -> bool:
    """
    Main entry point for preprocessing datasets.

    Args:
        data_dir: Directory containing raw downloaded data
        output_dir: Directory to save processed samples
        n_samples: Number of samples to extract
        seed: Random seed for reproducibility

    Returns:
        True if successful, False otherwise
    """
    random.seed(seed)

    print("=" * 60)
    print("TTS Evaluation Dataset Preprocessor")
    print("=" * 60)

    extractor = NavigationInstructionExtractor(data_dir, output_dir)

    try:
        # Extract from both datasets
        print("\n=== Extracting Touchdown Instructions ===")
        touchdown_instructions = extractor.extract_touchdown_instructions()
        print(f"✓ Extracted {len(touchdown_instructions)} Touchdown instructions")

        print("\n=== Extracting R2R Instructions ===")
        r2r_instructions = extractor.extract_r2r_instructions()
        print(f"✓ Extracted {len(r2r_instructions)} R2R instructions")

        # Combine
        all_instructions = touchdown_instructions + r2r_instructions
        print(f"\n✓ Total instructions: {len(all_instructions)}")

        # Filter by length and quality
        print("\n=== Filtering Instructions ===")
        filtered = extractor.filter_instructions(all_instructions)
        print(f"✓ After filtering: {len(filtered)} instructions")

        # Remove duplicates
        print("\n=== Removing Duplicates ===")
        unique = extractor.remove_duplicates(filtered)
        print(f"✓ After deduplication: {len(unique)} unique instructions")

        # Sample balanced dataset
        print(f"\n=== Sampling {n_samples} Instructions ===")
        sampled = extractor.balance_sources(unique, n_samples)
        print(f"✓ Sampled {len(sampled)} instructions")

        # Create final samples with IDs
        final_samples = extractor.create_final_samples(sampled)

        # Save to file
        print("\n=== Saving Samples ===")
        extractor.save_samples(final_samples)

        # Print statistics
        extractor.print_statistics(final_samples)

        print("\n" + "=" * 60)
        print("✓ Preprocessing completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_samples(samples_path: Optional[Path] = None) -> List[Dict]:
    """
    Load preprocessed samples from JSON file.

    Args:
        samples_path: Path to samples.json (defaults to data/samples.json)

    Returns:
        List of sample dictionaries
    """
    if samples_path is None:
        samples_path = Path(__file__).parent / "samples.json"

    with open(samples_path, 'r') as f:
        samples = json.load(f)

    return samples


if __name__ == "__main__":
    # Test preprocessing
    success = preprocess_datasets()
    exit(0 if success else 1)
