#!/bin/bash
#
# CNN Experiment Runner
# Easy launcher for memory-efficient CNN experiments
#
# Usage:
#   ./run_cnn_experiments.sh                    # Run all architectures
#   ./run_cnn_experiments.sh --arch CNN-2L      # Run only 2-layer CNN
#   ./run_cnn_experiments.sh --quick            # Quick test (1 trial)
#

# Configuration
DATA_PATH="/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/PreprocessedData1/csp_lda_data.npz"
OUTPUT_DIR="results_cnn_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=32
MAX_EPOCHS=50
PATIENCE=10
N_TRIALS=5

# Parse arguments
ARCHITECTURES="CNN-2L CNN-3L"
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCHITECTURES="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            N_TRIALS=1
            MAX_EPOCHS=10
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=============================================="
echo "CNN EXPERIMENT CONFIGURATION"
echo "=============================================="
echo "Data path:       $DATA_PATH"
echo "Output dir:      $OUTPUT_DIR"
echo "Architectures:   $ARCHITECTURES"
echo "Batch size:      $BATCH_SIZE"
echo "Max epochs:      $MAX_EPOCHS"
echo "Patience:        $PATIENCE"
echo "Trials:          $N_TRIALS"
if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo "⚠️  QUICK MODE ENABLED - Results not suitable for final report!"
fi
echo "=============================================="
echo ""

# Confirm
if [ "$QUICK_MODE" = false ]; then
    read -p "Start experiments? This may take several hours. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found: $DATA_PATH"
    echo ""
    echo "Please run preprocessing first:"
    echo "  python preprocessing_fixed.py --only cnn"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration
cat > "$OUTPUT_DIR/config.txt" << EOF
CNN Experiment Configuration
Generated: $(date)

Data path: $DATA_PATH
Architectures: $ARCHITECTURES
Batch size: $BATCH_SIZE
Max epochs: $MAX_EPOCHS
Patience: $PATIENCE
Trials: $N_TRIALS
Quick mode: $QUICK_MODE

Command:
python cnn_experiment.py \\
    --data-path "$DATA_PATH" \\
    --output-dir "$OUTPUT_DIR" \\
    --batch-size $BATCH_SIZE \\
    --max-epochs $MAX_EPOCHS \\
    --patience $PATIENCE \\
    --architectures $ARCHITECTURES \\
    --n-trials $N_TRIALS
EOF

# Start experiments
echo ""
echo "🚀 Starting CNN experiments..."
echo "   Logs will be saved to: $OUTPUT_DIR"
echo ""

python cnn_experiment.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --max-epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --architectures $ARCHITECTURES \
    --n-trials $N_TRIALS \
    2>&1 | tee "$OUTPUT_DIR/experiment_log.txt"

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiments completed successfully!"
    echo ""
    echo "Results location: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. View summary: cat $OUTPUT_DIR/aggregate_results.json"
    echo "  2. View plots: ls $OUTPUT_DIR/plots/"
    echo "  3. Analyze results: python analyze_cnn_results.py --results-dir $OUTPUT_DIR"
else
    echo ""
    echo "❌ Experiments failed. Check logs: $OUTPUT_DIR/experiment_log.txt"
    exit 1
fi