#!/bin/bash
set -e  # stop on first failure

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

# Merge incoming into training set
echo "Merging incoming data into training set..."
for csv in data/incoming/*.csv; do
    # Skip header line, append data rows
    tail -n +2 "$csv" >> data/training/train.csv
done

# Move processed files so they don't get picked up again
mkdir -p data/processed
mv data/incoming/*.csv data/processed/
echo "Merged and moved to data/processed/"

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

python scripts/evaluate.py \
    --new-checkpoint "$NEW_CHECKPOINT" \
    --prod-checkpoint "$PROD_CHECKPOINT" \
    --val-data data/validation/val.csv \
    --model ViT-B-32

# If evaluate.py exits non-zero, set -e stops the pipeline here

# -------------------------------------------------
# Step 5: Register in MLflow
# -------------------------------------------------
echo ""
echo "[5/5] Registering model..."

# Extract metrics from evaluate.py output (re-run with capture)
METRICS=$(python scripts/evaluate.py \
    --new-checkpoint "$NEW_CHECKPOINT" \
    --prod-checkpoint "$PROD_CHECKPOINT" \
    --val-data data/validation/val.csv \
    --model ViT-B-32 2>&1 | grep -oP '\{.*\}' || echo '{}')

python scripts/register.py \
    --checkpoint "$NEW_CHECKPOINT" \
    --metrics "$METRICS"

# Update production checkpoint
mkdir -p models/production
cp "$NEW_CHECKPOINT" "$PROD_CHECKPOINT"
echo "Updated production checkpoint"

echo ""
echo "============================================"
echo "  Pipeline complete — $RUN_NAME"
echo "============================================"
