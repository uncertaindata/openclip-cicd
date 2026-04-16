"""
Training wrapper — reads config and calls open_clip's training via torchrun.

Usage:
  python scripts/train.py --config configs/train_config.yaml
  python scripts/train.py --config configs/train_config.yaml --run-name experiment_v2
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train open_clip model")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--run-name", default=None, help="Name for this run (default: timestamp)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logs_dir = os.path.join(cfg["logs_dir"], run_name)

    cmd = [
        "torchrun",
        "--nproc_per_node", str(cfg["nproc"]),
        "-m", "open_clip_train.main",
        "--model", cfg["model"],
        "--pretrained", cfg["pretrained"],
        "--train-data", cfg["train_data"],
        "--val-data", cfg["val_data"],
        "--dataset-type", cfg["dataset_type"],
        "--batch-size", str(cfg["batch_size"]),
        "--epochs", str(cfg["epochs"]),
        "--lr", str(cfg["lr"]),
        "--warmup", str(cfg["warmup"]),
        "--workers", str(cfg["workers"]),
        "--precision", cfg["precision"],
        "--save-frequency", str(cfg["save_frequency"]),
        "--logs", logs_dir,
        "--report-to", cfg["report_to"],
        "--name", run_name,
    ]

    print(f"Starting training run: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Logs: {logs_dir}")
    print("-" * 60)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
