"""
Dataset download utilities for TTS evaluation.

Downloads Touchdown and Room-to-Room (R2R) navigation instruction datasets.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm


# Dataset URLs
TOUCHDOWN_URL = "https://github.com/lil-lab/touchdown/raw/master/data/train.json"
TOUCHDOWN_DEV_URL = "https://github.com/lil-lab/touchdown/raw/master/data/dev.json"

R2R_TRAIN_URL = "https://raw.githubusercontent.com/YicongHong/Fine-Grained-R2R/refs/heads/master/data/FGR2R_train.json"
R2R_VAL_SEEN_URL = "https://raw.githubusercontent.com/YicongHong/Fine-Grained-R2R/refs/heads/master/data/FGR2R_val_seen.json"


class DataDownloader:
    """Handles downloading and caching of navigation datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize downloader.

        Args:
            data_dir: Directory to store downloaded data. Defaults to ./data/raw/
        """
        if data_dir is None:
            # Get the data directory relative to this file
            data_dir = Path(__file__).parent / "raw"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, output_path: Path, desc: str = "Downloading") -> bool:
        """
        Download a file with progress bar.

        Args:
            url: URL to download from
            output_path: Local path to save file
            desc: Description for progress bar

        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"✓ Downloaded: {output_path.name}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {url}: {e}")
            return False

    def download_touchdown(self) -> bool:
        """
        Download Touchdown dataset.

        Returns:
            True if successful, False otherwise
        """
        print("\n=== Downloading Touchdown Dataset ===")

        touchdown_dir = self.data_dir / "touchdown"
        touchdown_dir.mkdir(exist_ok=True)

        train_path = touchdown_dir / "train.json"
        dev_path = touchdown_dir / "dev.json"

        success = True

        if not train_path.exists():
            success &= self.download_file(
                TOUCHDOWN_URL,
                train_path,
                "Touchdown train"
            )
        else:
            print(f"✓ Already exists: {train_path.name}")

        if not dev_path.exists():
            success &= self.download_file(
                TOUCHDOWN_DEV_URL,
                dev_path,
                "Touchdown dev"
            )
        else:
            print(f"✓ Already exists: {dev_path.name}")

        return success

    def download_r2r(self) -> bool:
        """
        Download Room-to-Room (R2R) dataset.

        Returns:
            True if successful, False otherwise
        """
        print("\n=== Downloading R2R Dataset ===")

        r2r_dir = self.data_dir / "r2r"
        r2r_dir.mkdir(exist_ok=True)

        train_path = r2r_dir / "R2R_train.json"
        val_path = r2r_dir / "R2R_val_seen.json"

        success = True

        if not train_path.exists():
            success &= self.download_file(
                R2R_TRAIN_URL,
                train_path,
                "R2R train"
            )
        else:
            print(f"✓ Already exists: {train_path.name}")

        if not val_path.exists():
            success &= self.download_file(
                R2R_VAL_SEEN_URL,
                val_path,
                "R2R val_seen"
            )
        else:
            print(f"✓ Already exists: {val_path.name}")

        return success

    def verify_downloads(self) -> bool:
        """
        Verify all required files exist and are valid JSON.

        Returns:
            True if all files valid, False otherwise
        """
        print("\n=== Verifying Downloads ===")

        required_files = [
            self.data_dir / "touchdown" / "train.json",
            self.data_dir / "touchdown" / "dev.json",
            self.data_dir / "r2r" / "R2R_train.json",
            self.data_dir / "r2r" / "R2R_val_seen.json",
        ]

        all_valid = True

        for file_path in required_files:
            if not file_path.exists():
                print(f"✗ Missing: {file_path.name}")
                all_valid = False
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = len(data.get('data', []))
                    else:
                        count = 0
                    print(f"✓ Valid: {file_path.name} ({count} entries)")
            except json.JSONDecodeError:
                print(f"✗ Invalid JSON: {file_path.name}")
                all_valid = False
            except Exception as e:
                print(f"✗ Error reading {file_path.name}: {e}")
                all_valid = False

        return all_valid


def download_datasets(data_dir: Optional[Path] = None) -> bool:
    """
    Main entry point for downloading all datasets.

    Args:
        data_dir: Directory to store downloaded data

    Returns:
        True if all downloads successful, False otherwise
    """
    downloader = DataDownloader(data_dir)

    print("=" * 60)
    print("TTS Evaluation Dataset Downloader")
    print("=" * 60)

    # Download datasets
    touchdown_success = downloader.download_touchdown()
    r2r_success = downloader.download_r2r()

    # Verify all downloads
    verification_success = downloader.verify_downloads()

    # Summary
    print("\n" + "=" * 60)
    if touchdown_success and r2r_success and verification_success:
        print("✓ All datasets downloaded and verified successfully!")
        print(f"✓ Data stored in: {downloader.data_dir}")
        return True
    else:
        print("✗ Some downloads failed. Please check errors above.")
        return False


if __name__ == "__main__":
    # Test download
    success = download_datasets()
    exit(0 if success else 1)
