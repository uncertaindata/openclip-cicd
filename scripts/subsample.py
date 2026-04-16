"""
Subsample all CSVs to N entries for quick testing.

Usage:
  python scripts/subsample.py --n 100
"""

import argparse
from pathlib import Path

import pandas as pd


def subsample(csv_path: Path, n: int):
    df = pd.read_csv(csv_path, sep="\t")
    original = len(df)
    if original <= n:
        print(f"  {csv_path}: {original} rows (unchanged, already <= {n})")
        return
    df = df.sample(n=n, random_state=42)
    df.to_csv(csv_path, sep="\t", index=False)
    print(f"  {csv_path}: {original} → {n} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    base = Path("data")
    csvs = list(base.rglob("*.csv"))

    print(f"Subsampling to {args.n} rows:")
    for csv in sorted(csvs):
        subsample(csv, args.n)
    print("Done.")


if __name__ == "__main__":
    main()
