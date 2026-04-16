"""
Validate incoming data before training.

Checks:
  1. CSV has required columns (filepath, title)
  2. CSV is tab-separated
  3. All image paths exist and are readable
  4. No duplicate entries
  5. Minimum sample count met

Usage:
  python scripts/validate_data.py --csv data/incoming/chunk_1.csv
  python scripts/validate_data.py --csv data/incoming/chunk_1.csv --min-samples 100
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


def validate_csv_structure(df: pd.DataFrame, csv_path: str) -> list[str]:
    """Check CSV has required columns."""
    errors = []
    required = {"filepath", "title"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"{csv_path}: missing columns {missing}. Found: {list(df.columns)}")
    return errors


def validate_images_exist(df: pd.DataFrame, csv_path: str) -> list[str]:
    """Check all referenced images exist on disk."""
    errors = []
    missing_count = 0
    for path in df["filepath"]:
        if not Path(path).exists():
            missing_count += 1
            if missing_count <= 5:
                errors.append(f"{csv_path}: image not found: {path}")
    if missing_count > 5:
        errors.append(f"{csv_path}: ... and {missing_count - 5} more missing images")
    return errors


def validate_images_readable(df: pd.DataFrame, csv_path: str, sample_n: int = 50) -> list[str]:
    """Spot-check that a sample of images can actually be opened."""
    errors = []
    sample = df["filepath"].sample(n=min(sample_n, len(df)), random_state=42)
    corrupt_count = 0
    for path in sample:
        try:
            img = Image.open(path)
            img.verify()
        except Exception as e:
            corrupt_count += 1
            if corrupt_count <= 3:
                errors.append(f"{csv_path}: corrupt image: {path} ({e})")
    if corrupt_count > 3:
        errors.append(f"{csv_path}: ... and {corrupt_count - 3} more corrupt images in sample")
    return errors


def validate_duplicates(df: pd.DataFrame, csv_path: str) -> list[str]:
    """Check for duplicate filepaths. Warns but does not fail."""
    warnings = []
    dupes = df["filepath"].duplicated().sum()
    if dupes > 0:
        print(f"  WARNING: {csv_path}: {dupes} duplicate filepath entries (will be dropped)")
        df.drop_duplicates(subset="filepath", inplace=True)
    return warnings  # returns empty — duplicates are not fatal


def validate_min_samples(df: pd.DataFrame, csv_path: str, min_samples: int) -> list[str]:
    """Check minimum sample count."""
    errors = []
    if len(df) < min_samples:
        errors.append(f"{csv_path}: only {len(df)} samples, need at least {min_samples}")
    return errors


def validate(csv_path: str, min_samples: int = 10) -> list[str]:
    """Run all validations on a CSV file. Returns list of error strings."""
    try:
        df = pd.read_csv(csv_path, sep="\t")
    except Exception as e:
        return [f"{csv_path}: failed to read CSV: {e}"]

    errors = []
    errors += validate_csv_structure(df, csv_path)
    if errors:
        return errors  # can't continue without correct columns

    errors += validate_min_samples(df, csv_path, min_samples)
    errors += validate_duplicates(df, csv_path)
    errors += validate_images_exist(df, csv_path)
    errors += validate_images_readable(df, csv_path)

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate incoming training data")
    parser.add_argument("--csv", required=True, help="Path to CSV file to validate")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum required samples")
    args = parser.parse_args()

    print(f"Validating {args.csv} ...")
    errors = validate(args.csv, args.min_samples)

    if errors:
        print(f"\nFAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"PASSED — {args.csv} is valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
