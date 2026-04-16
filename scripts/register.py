"""
Register a trained model in MLflow model registry.

Usage:
  python scripts/register.py \
      --checkpoint logs/run_xxx/checkpoints/epoch_5.pt \
      --model-name openclip-vit-b-32 \
      --metrics '{"recall@1": 0.45, "recall@5": 0.78}' \
      --stage staging
"""

import argparse
import json

import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = "http://192.168.1.61:5000"


def main():
    parser = argparse.ArgumentParser(description="Register model in MLflow")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-name", default="openclip-vit-b-32", help="Registry model name")
    parser.add_argument("--metrics", required=True, help="JSON string of evaluation metrics")
    parser.add_argument("--stage", default="staging", choices=["staging", "production"],
                        help="Model stage to assign")
    args = parser.parse_args()

    metrics = json.loads(args.metrics)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"register_{args.model_name}"):
        # Log metrics
        for key, val in metrics.items():
            safe_key = key.replace("@", "_at_")
            mlflow.log_metric(safe_key, val)

        # Log the checkpoint as an artifact
        mlflow.log_artifact(args.checkpoint)

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{args.checkpoint}"
        result = mlflow.register_model(model_uri, args.model_name)

        # Set stage
        client.transition_model_version_stage(
            name=args.model_name,
            version=result.version,
            stage=args.stage,
        )

        print(f"Registered {args.model_name} v{result.version} as '{args.stage}'")
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
