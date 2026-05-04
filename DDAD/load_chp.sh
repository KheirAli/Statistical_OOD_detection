#!/bin/bash
# run_all_categories.sh

CONFIG="config.yaml"
GPU="cuda"

# Format: "category:load_chp"
CATEGORIES=(
    # "bottle:1000"
    "cable:3000"
    "capsule:1500"
    "carpet:2500"
    "grid:2000"
    "hazelnut:2000"
    "leather:2000"
    "metal_nut:3000"
    "pill:1000"
    "screw:2000"
    "tile:1000"
    "toothbrush:2000"
    "transistor:2000"
    "wood:2000"
    "zipper:1000"
)

RESULTS_FILE="results_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "DDAD Detection Results" > "$RESULTS_FILE"
echo "======================" >> "$RESULTS_FILE"

for ENTRY in "${CATEGORIES[@]}"; do
    CATEGORY="${ENTRY%%:*}"
    LOAD_CHP="${ENTRY##*:}"

    echo ""
    echo "======================================"
    echo "Category: ${CATEGORY}  |  Checkpoint: ${LOAD_CHP}"
    echo "======================================"

    python main.py \
        --config   "$CONFIG" \
        --detection True \
        --category "$CATEGORY" \
        --load_chp "$LOAD_CHP" \
        2>&1 | tee -a "$RESULTS_FILE"

    echo "Done: ${CATEGORY}" | tee -a "$RESULTS_FILE"
done

echo ""
echo "All done. Results saved to: $RESULTS_FILE"