"""
Evaluation gate — compare newly trained model against production model.

Loads both models, runs them on the validation set, compares retrieval metrics.
Exits 0 if new model is better (pipeline continues), exits 1 if not (pipeline stops).

Usage:
  python scripts/evaluate.py \
      --new-checkpoint logs/run_xxx/checkpoints/epoch_5.pt \
      --prod-checkpoint models/production/epoch_latest.pt \
      --val-data data/validation/val.csv \
      --model ViT-B-32 \
      --threshold 0.005
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip


def load_model(model_name: str, checkpoint_path: str, device: str):
    """Load an open_clip model from checkpoint."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=checkpoint_path
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def compute_retrieval_metrics(model, preprocess, tokenizer, val_df, device: str):
    """
    Compute image-text retrieval recall@1 and recall@5.
    For each image, rank all captions by similarity — check if the correct caption is in top-k.
    """
    image_features_list = []
    text_features_list = []

    with torch.no_grad(), torch.amp.autocast(device):
        # Encode all images
        for path in tqdm(val_df["filepath"], desc="Encoding images"):
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            image_features_list.append(feat)

        # Encode all captions
        for caption in tqdm(val_df["title"], desc="Encoding text"):
            text = tokenizer([caption]).to(device)
            feat = model.encode_text(text)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            text_features_list.append(feat)

    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)

    # Similarity matrix: (num_images x num_texts)
    similarity = (image_features @ text_features.T).cpu()

    # Image-to-text retrieval: for each image, rank texts
    n = len(similarity)
    ranks = similarity.argsort(dim=1, descending=True)
    correct = torch.arange(n).unsqueeze(1)
    match_positions = (ranks == correct).nonzero(as_tuple=True)[1]

    recall_at_1 = (match_positions < 1).float().mean().item()
    recall_at_5 = (match_positions < 5).float().mean().item()

    return {"recall@1": recall_at_1, "recall@5": recall_at_5}


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--new-checkpoint", required=True, help="Path to newly trained checkpoint")
    parser.add_argument("--prod-checkpoint", required=True, help="Path to current production checkpoint")
    parser.add_argument("--val-data", required=True, help="Path to validation CSV")
    parser.add_argument("--model", default="ViT-B-32", help="Model architecture name")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="New model must beat prod by this margin on recall@1")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max validation samples to use (for speed)")
    parser.add_argument("--metrics-out", default="metrics.json",
                        help="Path to save metrics JSON (used by register.py)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_df = pd.read_csv(args.val_data, sep="\t")
    if len(val_df) > args.max_samples:
        val_df = val_df.sample(n=args.max_samples, random_state=42)

    print(f"Evaluating on {len(val_df)} validation samples")
    print(f"Device: {device}")
    print("-" * 60)

    # Evaluate new model
    print(f"\nLoading new model: {args.new_checkpoint}")
    new_model, new_preprocess, new_tokenizer = load_model(args.model, args.new_checkpoint, device)
    new_metrics = compute_retrieval_metrics(new_model, new_preprocess, new_tokenizer, val_df, device)
    del new_model
    torch.cuda.empty_cache()

    # First run — no production model yet
    if not os.path.exists(args.prod_checkpoint):
        print(f"\nNo production model found at {args.prod_checkpoint}")
        print("First run — skipping comparison, auto-promoting new model")
        print(f"\nNew model metrics:")
        for key, val in new_metrics.items():
            print(f"  {key}: {val:.4f}")

        # Save metrics
        with open(args.metrics_out, "w") as f:
            json.dump(new_metrics, f)
        print(f"\nMetrics saved to {args.metrics_out}")
        sys.exit(0)

    # Evaluate production model
    print(f"\nLoading production model: {args.prod_checkpoint}")
    prod_model, prod_preprocess, prod_tokenizer = load_model(args.model, args.prod_checkpoint, device)
    prod_metrics = compute_retrieval_metrics(prod_model, prod_preprocess, prod_tokenizer, val_df, device)
    del prod_model
    torch.cuda.empty_cache()

    # Compare
    print("\n" + "=" * 60)
    print(f"{'Metric':<12} {'Production':>12} {'New':>12} {'Delta':>12}")
    print("-" * 60)
    for key in prod_metrics:
        delta = new_metrics[key] - prod_metrics[key]
        print(f"{key:<12} {prod_metrics[key]:>12.4f} {new_metrics[key]:>12.4f} {delta:>+12.4f}")
    print("=" * 60)

    # Save metrics
    with open(args.metrics_out, "w") as f:
        json.dump(new_metrics, f)
    print(f"Metrics saved to {args.metrics_out}")

    # Gate decision
    delta_r1 = new_metrics["recall@1"] - prod_metrics["recall@1"]
    if delta_r1 >= args.threshold:
        print(f"\nPASSED — new model beats production by {delta_r1:+.4f} (threshold: {args.threshold})")
        sys.exit(0)
    else:
        print(f"\nFAILED — new model delta {delta_r1:+.4f} below threshold {args.threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
