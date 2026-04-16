#!/bin/bash
set -e  # stop on first failure

# On failure, restore original training set
cleanup_on_failure() {
    if [ -f data/training/train.csv.bak ]; then
        echo "Pipeline failed — restoring original training set"
        mv data/training/train.csv.bak data/training/train.csv
    fi
}
trap cleanup_on_failure ERR

WORKSPACE="/workspace"
CONFIG="configs/train_config.yaml"
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
PROD_CHECKPOINT="models/production/latest.pt"

echo "============================================"
echo "  ML CI/CD Pipeline — $RUN_NAME"
echo "============================================"

# -------------------------------------------------
# Step 1: Check for new data in incoming/
# -------------------------------------------------
echo ""
echo "[1/5] Checking for new data..."

INCOMING_FILES=$(ls data/incoming/*.csv 2>/dev/null || true)
if [ -z "$INCOMING_FILES" ]; then
    echo "No new data in data/incoming/. Nothing to do."
    exit 0
fi
echo "Found: $INCOMING_FILES"

# -------------------------------------------------
# Step 2: Validate incoming data
# -------------------------------------------------
echo ""
echo "[2/5] Validating incoming data..."

for csv in data/incoming/*.csv; do
    python scripts/validate_data.py --csv "$csv"
done

# Merge incoming into training set (keep originals in incoming/ until pipeline succeeds)
echo "Merging incoming data into training set..."
cp data/training/train.csv data/training/train.csv.bak
for csv in data/incoming/*.csv; do
    tail -n +2 "$csv" >> data/training/train.csv
done
echo "Merged. Originals stay in incoming/ until pipeline completes."

# -------------------------------------------------
# Step 3: Train
# -------------------------------------------------
echo ""
echo "[3/5] Training..."

python scripts/train.py --config "$CONFIG" --run-name "$RUN_NAME"

# Find the latest checkpoint
NEW_CHECKPOINT=$(find "logs/$RUN_NAME" -name "epoch_*.pt" | sort -V | tail -1)
echo "New checkpoint: $NEW_CHECKPOINT"

# -------------------------------------------------
# Step 4: Evaluate (the gate)
# -------------------------------------------------
echo ""
echo "[4/5] Evaluating new model vs production..."

METRICS_FILE="logs/$RUN_NAME/metrics.json"

python scripts/evaluate.py \
    --new-checkpoint "$NEW_CHECKPOINT" \
    --prod-checkpoint "$PROD_CHECKPOINT" \
    --val-data data/validation/val.csv \
    --model ViT-B-32 \
    --metrics-out "$METRICS_FILE"

# If evaluate.py exits non-zero, set -e stops the pipeline here

# -------------------------------------------------
# Step 5: Register in MLflow
# -------------------------------------------------
echo ""
echo "[5/5] Registering model..."

python scripts/register.py \
    --checkpoint "$NEW_CHECKPOINT" \
    --metrics "$(cat $METRICS_FILE)"

# Update production checkpoint
mkdir -p models/production
cp "$NEW_CHECKPOINT" "$PROD_CHECKPOINT"
echo "Updated production checkpoint"

# Move processed files now that everything succeeded
mkdir -p data/processed
mv data/incoming/*.csv data/processed/
rm -f data/training/train.csv.bak
echo "Moved incoming data to data/processed/"

echo ""
echo "============================================"
echo "  Pipeline complete — $RUN_NAME"
echo "============================================"
