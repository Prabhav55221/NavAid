"""
MOS (Mean Opinion Score) sample selection.

Selects representative audio samples for human rating.
"""

import random
from pathlib import Path
from typing import List, Dict
import shutil
import csv


def select_mos_samples(all_samples: List[Dict], n_samples: int = 30,
                      samples_per_model: int = 6, seed: int = 42) -> Dict[str, List[Dict]]:
    """
    Select samples for MOS rating, balanced across models.

    Args:
        all_samples: List of all generated samples with metadata
                    Format: [{'model': 'X', 'sample_id': 'Y', 'audio_path': 'Z', 'text': 'T'}, ...]
        n_samples: Total number of samples to select
        samples_per_model: Samples per model (should equal n_samples / num_models)
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping model names to selected samples
    """
    random.seed(seed)

    # Group by model
    by_model = {}
    for sample in all_samples:
        model = sample['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(sample)

    # Select samples per model
    selected = {}

    for model, samples in by_model.items():
        # Randomly sample
        n_select = min(samples_per_model, len(samples))
        selected[model] = random.sample(samples, n_select)

    return selected


def copy_mos_samples(selected_samples: Dict[str, List[Dict]],
                    output_dir: Path) -> List[Dict]:
    """
    Copy selected samples to MOS directory and create rating sheet.

    Args:
        selected_samples: Dictionary of selected samples per model
        output_dir: Output directory for MOS samples

    Returns:
        List of all samples with new paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_mos_samples = []
    sample_counter = 1

    for model, samples in selected_samples.items():
        model_dir = output_dir / model
        model_dir.mkdir(exist_ok=True)

        for sample in samples:
            # Create new filename
            original_path = Path(sample['audio_path'])
            new_filename = f"mos_{sample_counter:03d}_{model}_{original_path.stem}.wav"
            new_path = model_dir / new_filename

            # Copy audio file
            try:
                shutil.copy2(original_path, new_path)

                all_mos_samples.append({
                    'mos_id': sample_counter,
                    'model': model,
                    'sample_id': sample['sample_id'],
                    'text': sample['text'],
                    'audio_path': str(new_path),
                    'filename': new_filename
                })

                sample_counter += 1

            except Exception as e:
                print(f"Warning: Failed to copy {original_path}: {e}")

    return all_mos_samples


def create_mos_rating_sheet(mos_samples: List[Dict], output_path: Path):
    """
    Create CSV rating sheet for MOS evaluation.

    Args:
        mos_samples: List of MOS samples
        output_path: Path to output CSV file
    """
    # Randomize order for rating (hide model identity)
    random.shuffle(mos_samples)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Sample_ID',
            'Filename',
            'Text',
            'Naturalness_1_5',
            'Intelligibility_1_5',
            'Overall_Quality_1_5',
            'Comments',
            'Rater_Name'
        ])

        # Rows for each sample
        for sample in mos_samples:
            writer.writerow([
                sample['mos_id'],
                sample['filename'],
                sample['text'],
                '',  # Naturalness rating (to be filled)
                '',  # Intelligibility rating
                '',  # Overall quality rating
                '',  # Comments
                ''   # Rater name
            ])

    print(f"✓ Created MOS rating sheet: {output_path}")
    print(f"  Total samples: {len(mos_samples)}")
    print(f"\nInstructions for raters:")
    print("  1. Listen to each audio file")
    print("  2. Rate on 1-5 scale:")
    print("     1 = Very poor, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent")
    print("  3. Naturalness: How human-like does it sound?")
    print("  4. Intelligibility: How easy is it to understand?")
    print("  5. Overall Quality: General impression")
    print("  6. Add any comments if desired")
    print("  7. Enter your name in Rater_Name column")


def create_mos_instructions(output_dir: Path):
    """Create instructions file for MOS raters."""
    instructions_path = output_dir / "RATING_INSTRUCTIONS.txt"

    instructions = """
MOS (Mean Opinion Score) RATING INSTRUCTIONS
============================================

Thank you for participating in this TTS evaluation!

OVERVIEW:
You will listen to and rate 30 short navigation instruction audio samples.
Each sample is synthesized by one of 5 different text-to-speech systems.

TASK:
1. Open the file: mos_rating_sheet.csv
2. For each row:
   a. Find the audio file in the corresponding model subfolder
   b. Listen to the audio (you can replay it multiple times)
   c. Rate the audio on three 1-5 scales:
      - Naturalness: How human-like does the voice sound?
      - Intelligibility: How easy is it to understand the words?
      - Overall Quality: Your general impression

RATING SCALE:
5 = Excellent (very natural, perfectly clear, high quality)
4 = Good (mostly natural, clear, good quality)
3 = Fair (somewhat natural, understandable, acceptable quality)
2 = Poor (robotic, difficult to understand, low quality)
1 = Very Poor (unnatural, unintelligible, very low quality)

GUIDELINES:
- Rate based on your first impression after 1-2 listens
- Consider the context: these are navigation instructions for visually impaired users
- Intelligibility is most important for safety-critical use
- Don't worry if you hear the same text multiple times (different systems)
- Add comments if a sample has unusual characteristics

OPTIONAL COMMENTS:
Feel free to note:
- Specific pronunciation issues
- Background noise or artifacts
- Pacing problems (too fast/slow)
- Any other observations

When finished, save the CSV and return it to the research team.

Questions? Contact the team.

Thank you!
"""

    instructions_path.write_text(instructions)
    print(f"✓ Created rating instructions: {instructions_path}")


def select_and_prepare_mos(outputs_dir: Path, samples_data: List[Dict],
                           mos_output_dir: Path, n_samples: int = 30) -> bool:
    """
    Complete MOS preparation workflow.

    Args:
        outputs_dir: Directory with all generated audio files
        samples_data: Metadata for all samples
        mos_output_dir: Output directory for MOS samples
        n_samples: Total number of samples to select

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("MOS SAMPLE SELECTION AND PREPARATION")
    print("=" * 60)

    # Select samples
    print(f"\nSelecting {n_samples} samples for MOS rating...")
    samples_per_model = n_samples // len(set(s['model'] for s in samples_data))

    selected = select_mos_samples(
        samples_data,
        n_samples=n_samples,
        samples_per_model=samples_per_model
    )

    print(f"✓ Selected samples:")
    for model, samples in selected.items():
        print(f"  {model}: {len(samples)} samples")

    # Copy samples to MOS directory
    print(f"\nCopying samples to {mos_output_dir}...")
    mos_samples = copy_mos_samples(selected, mos_output_dir)

    print(f"✓ Copied {len(mos_samples)} audio files")

    # Create rating sheet
    rating_sheet_path = mos_output_dir / "mos_rating_sheet.csv"
    create_mos_rating_sheet(mos_samples, rating_sheet_path)

    # Create instructions
    create_mos_instructions(mos_output_dir)

    print("\n" + "=" * 60)
    print("✓ MOS PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Share these files with 4 raters:")
    print(f"   - {mos_output_dir}")
    print(f"   - Contains audio files organized by model")
    print(f"   - Rating sheet (mos_rating_sheet.csv)")
    print(f"   - Instructions (RATING_INSTRUCTIONS.txt)")
    print(f"2. Raters fill out the CSV")
    print(f"3. Collect completed CSVs and compute MOS scores")

    return True
