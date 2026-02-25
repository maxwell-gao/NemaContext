#!/bin/bash
# Monitor training progress and auto-run discovery when complete

LOG_FILE="logs/crossmodal_100ep.log"
CKPT_DIR="checkpoints_trimodal_crossmodal"
DISCOVERY_DIR="discoveries"

echo "=== NemaContext Training Monitor ==="
echo "Monitoring: $LOG_FILE"
echo ""

# Check if training is running
if pgrep -f "train_trimodal_crossmodal.py" > /dev/null; then
    echo "Status: TRAINING IN PROGRESS"
    echo ""

    # Show latest progress
    if [ -f "$LOG_FILE" ]; then
        echo "Latest epochs:"
        tail -10 "$LOG_FILE" | grep "Epoch"
    fi

    echo ""
    echo "Checkpoint sizes:"
    ls -lh "$CKPT_DIR"/ 2>/dev/null || echo "  No checkpoints yet"
else
    echo "Status: TRAINING COMPLETE"
    echo ""

    # Run discovery if not already done
    if [ ! -d "$DISCOVERY_DIR" ]; then
        echo "Running discovery pipeline..."

        uv run python examples/discover_priors.py \
            --checkpoint "$CKPT_DIR/best.pt" \
            --output "$DISCOVERY_DIR" \
            --n_samples 100

        echo ""
        echo "Running modality completion evaluation..."

        uv run python examples/evaluate_modality_completion.py \
            --checkpoint "$CKPT_DIR/best.pt" \
            --test_mode gene_to_spatial \
            --n_test 100
    else
        echo "Discovery already completed. Results in: $DISCOVERY_DIR"
    fi
fi

echo ""
echo "=== Monitor Complete ==="
