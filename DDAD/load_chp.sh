# #!/bin/bash
# # run_all_categories.sh

# CONFIG="config.yaml"
# GPU="cuda"

# # Format: "category:load_chp"
# CATEGORIES=(
#     "CT:2000"
#     # "faces:2000"
#     # "bottle:1000"
#     # "cable:3000"
#     # "capsule:1500"
#     # "carpet:2500"
#     # "grid:2000"
#     # "hazelnut:2000"
#     # "leather:2000"
#     # "metal_nut:3000"
#     # "pill:1000"
#     # "screw:2000"
#     # "tile:1000"
#     # "toothbrush:2000"
#     # "transistor:2000"
#     # "wood:2000"
#     # "zipper:1000"
# )

# RESULTS_FILE="results_summary_$(date +%Y%m%d_%H%M%S).txt"
# echo "DDAD Detection Results" > "$RESULTS_FILE"
# echo "======================" >> "$RESULTS_FILE"

# for ENTRY in "${CATEGORIES[@]}"; do
#     CATEGORY="${ENTRY%%:*}"
#     LOAD_CHP="${ENTRY##*:}"

#     echo ""
#     echo "======================================"
#     echo "Category: ${CATEGORY}  |  Checkpoint: ${LOAD_CHP}"
#     echo "======================================"

#     python main.py \
#         --config   "$CONFIG" \
#         --detection True \
#         --category "$CATEGORY" \
#         --load_chp "$LOAD_CHP" \
#         2>&1 | tee -a "$RESULTS_FILE"

#     echo "Done: ${CATEGORY}" | tee -a "$RESULTS_FILE"
# done

# echo ""
# echo "All done. Results saved to: $RESULTS_FILE"


#!/bin/bash
# run_all_categories.sh
#!/bin/bash
# run_all_categories.sh

# CONFIG="config.yaml"

# # ── Seed pools ────────────────────────────────────────────────────────────────
# # All 20 seeds — subsets are taken from the front of this list:
# #   N=1  → (42)
# #   N=10 → (42 123 456 789 1337 2024 99 7 314 2718)
# #   N=20 → all of them
# ALL_SEEDS=(42 123 456 789 1337 2024 99 7 314 2718 555 888 1000 2000 3000 4000 5000 6000 7000 8000)
# SEED_COUNTS=(20)      # runs three separate averaging experiments
# # ─────────────────────────────────────────────────────────────────────────────

# # CATEGORIES=(
# #     # "CT:2000"
# #     "faces:2000"
# #     "bottle:1000"
# #     "cable:3000"
# #     "capsule:1500"
# #     "carpet:2500"
# #     "grid:2000"
# #     "hazelnut:2000"
# #     "leather:2000"
# #     "metal_nut:3000"
# #     "pill:1000"
# #     "screw:2000"
# #     "tile:1000"
# #     "toothbrush:2000"
# #     "transistor:2000"
# #     "wood:2000"
# #     "zipper:1000"
# # )
# CATEGORIES=(
#     "CT:2000:combined"
#     "faces:2000:faces"       # lives under MVTec/faces/faces/2000
#     "bottle:1000:combined"
#     "cable:3000:combined"
#     "capsule:1500:combined"
#     "carpet:2500:combined"
#     "grid:2000:combined"
#     "hazelnut:2000:combined"
#     "leather:2000:combined"
#     "metal_nut:3000:combined"
#     "pill:1000:combined"
#     "screw:2000:combined"
#     "tile:1000:combined"
#     "toothbrush:2000:combined"
#     "transistor:2000:combined"
#     "wood:2000:combined"
#     "zipper:1000:combined"
# )

# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# FULL_LOG="results_full_${TIMESTAMP}.txt"
# AVG_FILE="results_averaged_${TIMESTAMP}.txt"

# echo "DDAD Detection — Full Log"   >  "$FULL_LOG"
# echo "DDAD Detection — Averaged Results" >  "$AVG_FILE"
# echo "Timestamp: $TIMESTAMP"            >> "$AVG_FILE"
# echo ""                                 >> "$AVG_FILE"

# # ── Helper: compute mean ± std from a list of floats ─────────────────────────
# stats() {
#     # usage: stats val1 val2 ...  →  prints "mean std"
#     python3 - "$@" <<'EOF'
# import sys, math
# vals = list(map(float, sys.argv[1:]))
# n = len(vals)
# mean = sum(vals) / n
# std  = math.sqrt(sum((x - mean)**2 for x in vals) / (n - 1)) if n > 1 else 0.0
# print(f"{mean:.4f} {std:.4f}")
# EOF
# }
# export -f stats
# # ─────────────────────────────────────────────────────────────────────────────

# for ENTRY in "${CATEGORIES[@]}"; do
#     # CATEGORY="${ENTRY%%:*}"
#     # LOAD_CHP="${ENTRY##*:}"

#     # echo ""
#     # echo "######################################"
#     # echo "#  Category: ${CATEGORY}  |  Checkpoint: ${LOAD_CHP}"
#     # echo "######################################"
#     CATEGORY="${ENTRY%%:*}"
#     REST="${ENTRY#*:}"
#     LOAD_CHP="${REST%%:*}"
#     SUBDIR="${REST##*:}"
#     # If no subdir was specified, default to "combined"
#     if [ "$SUBDIR" = "$LOAD_CHP" ]; then
#         SUBDIR="combined"
#     fi

#     CKPT_PATH="${CHECKPOINT_BASE}/${SUBDIR}/${CATEGORY}/${LOAD_CHP}"

#     echo ""
#     echo "======================================"
#     echo "Category: ${CATEGORY} | Checkpoint: ${LOAD_CHP} | Path: ${CKPT_PATH}"
#     echo "======================================"

#     # Guard: skip entirely if checkpoint is missing
#     if [ ! -e "$CKPT_PATH" ]; then
#         echo "[SKIP] Checkpoint not found: ${CKPT_PATH}" | tee -a "$FULL_LOG"
#         continue
#     fi

#     # ── Outer loop: run the same category for each seed-count experiment ──
#     for N in "${SEED_COUNTS[@]}"; do
#         SEEDS=("${ALL_SEEDS[@]:0:$N}")          # first N seeds from the pool

#         echo ""
#         echo "  === Averaging over N=${N} seeds: ${SEEDS[*]} ==="
#         echo "" >> "$AVG_FILE"
#         echo "Category: ${CATEGORY} | Checkpoint: ${LOAD_CHP} | N_seeds: ${N}" >> "$AVG_FILE"

#         # Collect per-seed metric values:  metric_name -> array of floats
#         declare -A METRIC_VALS   # e.g.  METRIC_VALS[image_auroc]="0.91 0.89 ..."

#         for SEED in "${SEEDS[@]}"; do
#             echo ""
#             echo "    -- Seed ${SEED} (N=${N}) --"

#             SEED_OUT=$(python main.py \
#                 --config    "$CONFIG" \
#                 --detection True \
#                 --category  "$CATEGORY" \
#                 --load_chp  "$LOAD_CHP" \
#                 --checkpoint_dir "${CHECKPOINT_BASE}/${SUBDIR}" \
#                 --seed      "$SEED" \
#                 2>&1 | tee -a "$FULL_LOG")

#             echo "    Done seed ${SEED}" | tee -a "$FULL_LOG"

#             # ── Parse metric lines from this seed's output ────────────────
#             # Expects lines like:  "image_auroc: 0.9123"  or  "AUROC 0.91"
#             # Adjust the grep/awk pattern to match your actual output format.
#             # while IFS= read -r line; do
#             #     # Normalize: replace common separators so awk sees "key value"
#             #     normalized=$(echo "$line" | sed 's/[=:]/ /g')
#             #     key=$(echo "$normalized"  | awk '{print tolower($1)}')
#             #     val=$(echo "$normalized"  | awk '{print $2}')
#             #     if [[ "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
#             #         METRIC_VALS[$key]="${METRIC_VALS[$key]} $val"
#             #     fi
#             # done < <(echo "$SEED_OUT" | grep -iE "auroc|aupro|f1|ap|pro|score")
#             while IFS= read -r line; do
#                 if [[ "$line" =~ "Per-image Pixel AUROC (mean):" ]]; then
#                     val=$(echo "$line" | awk -F': ' '{print $2}' | tr -d ' ')
#                     METRIC_VALS[pixel_auroc]="${METRIC_VALS[pixel_auroc]} $val"
#                 elif [[ "$line" =~ ^"SNR:" ]]; then
#                     val=$(echo "$line" | awk '{print $2}')
#                     METRIC_VALS[snr]="${METRIC_VALS[snr]} $val"
#                 fi
#             done <<< "$SEED_OUT"
#             # ─────────────────────────────────────────────────────────────
#         done

#         # ── Print mean ± std for every collected metric ───────────────────
#         for metric in "${!METRIC_VALS[@]}"; do
#             vals_arr=(${METRIC_VALS[$metric]})          # split on spaces
#             actual_n=${#vals_arr[@]}
#             result=$(stats "${vals_arr[@]}")
#             mean_v=$(echo "$result" | awk '{print $1}')
#             std_v=$(echo  "$result" | awk '{print $2}')
#             line="  $(printf '%-22s' "$metric")  mean=${mean_v}  std=${std_v}  (n=${actual_n})"
#             echo "$line"
#             echo "$line" >> "$AVG_FILE"
#         done

#         unset METRIC_VALS
#         declare -A METRIC_VALS    # reset for next N
#         echo "  ---" >> "$AVG_FILE"
#     done
# done

# echo ""
# echo "All done."
# echo "  Full log : $FULL_LOG"
# echo "  Averages : $AVG_FILE"


#!/bin/bash
# run_all_categories.sh

# CONFIG="config.yaml"

# # ── Correct base paths ────────────────────────────────────────────────────────
# DDAD_CKPT_BASE="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec"
# MVTEC_CKPT_BASE="/data/akherandish3/MVTec"
# # ─────────────────────────────────────────────────────────────────────────────

# ALL_SEEDS=(42 123 456 789 1337 2024 99 7 314 2718 555 888 1000 2000 3000 4000 5000 6000 7000 8000)
# SEED_COUNTS=(20)

# # Format: "category:load_chp:base_path"
# # base_path is the directory that contains  <category>/<checkpoint>
# CATEGORIES=(
#     # "CT:2000:${DDAD_CKPT_BASE}/combined"
#     "faces:2000:${DDAD_CKPT_BASE}/faces"    # checkpoint is at .../faces/faces/2000 → base is .../faces/faces
#     "bottle:1000:${MVTEC_CKPT_BASE}"
#     "cable:3000:${MVTEC_CKPT_BASE}"
#     "capsule:1500:${MVTEC_CKPT_BASE}"
#     "carpet:2500:${MVTEC_CKPT_BASE}"
#     "grid:2000:${MVTEC_CKPT_BASE}"
#     "hazelnut:2000:${MVTEC_CKPT_BASE}"
#     "leather:2000:${MVTEC_CKPT_BASE}"
#     "metal_nut:3000:${MVTEC_CKPT_BASE}"
#     "pill:1000:${MVTEC_CKPT_BASE}"
#     "screw:2000:${MVTEC_CKPT_BASE}"
#     "tile:1000:${MVTEC_CKPT_BASE}"
#     "toothbrush:2000:${MVTEC_CKPT_BASE}"
#     "transistor:2000:${MVTEC_CKPT_BASE}"
#     "wood:2000:${MVTEC_CKPT_BASE}"
#     "zipper:1000:${MVTEC_CKPT_BASE}"
# )

# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# FULL_LOG="results_full_${TIMESTAMP}.txt"
# AVG_FILE="results_averaged_${TIMESTAMP}.txt"

# echo "DDAD Detection — Full Log"          >  "$FULL_LOG"
# echo "DDAD Detection — Averaged Results"  >  "$AVG_FILE"
# echo "Timestamp: $TIMESTAMP"              >> "$AVG_FILE"
# echo ""                                   >> "$AVG_FILE"

# stats() {
#     python3 - "$@" <<'EOF'
# import sys, math
# vals = list(map(float, sys.argv[1:]))
# n = len(vals)
# mean = sum(vals) / n
# std  = math.sqrt(sum((x - mean)**2 for x in vals) / (n - 1)) if n > 1 else 0.0
# print(f"{mean:.4f} {std:.4f}")
# EOF
# }
# export -f stats

# for ENTRY in "${CATEGORIES[@]}"; do
#     # Parse "category:chp:base_path"  (base_path may contain colons → cut from right)
#     CATEGORY=$(echo "$ENTRY" | cut -d: -f1)
#     LOAD_CHP=$(echo  "$ENTRY" | cut -d: -f2)
#     BASE_PATH=$(echo "$ENTRY" | cut -d: -f3-)

#     CKPT_PATH="${BASE_PATH}/${CATEGORY}/${LOAD_CHP}"

#     echo ""
#     echo "======================================"
#     echo "Category: ${CATEGORY} | Checkpoint: ${LOAD_CHP}"
#     echo "Path: ${CKPT_PATH}"
#     echo "======================================"

#     # Skip immediately if checkpoint is missing — print the resolved path clearly
#     if [ ! -e "$CKPT_PATH" ]; then
#         echo "[SKIP] Not found: ${CKPT_PATH}" | tee -a "$FULL_LOG"
#         echo "[SKIP] ${CATEGORY} — checkpoint not found: ${CKPT_PATH}" >> "$AVG_FILE"
#         continue
#     fi

#     for N in "${SEED_COUNTS[@]}"; do
#         SEEDS=("${ALL_SEEDS[@]:0:$N}")

#         echo ""
#         echo "  === N=${N} seeds: ${SEEDS[*]} ==="
#         echo "" >> "$AVG_FILE"
#         echo "Category: ${CATEGORY} | Checkpoint: ${LOAD_CHP} | N_seeds: ${N}" >> "$AVG_FILE"

#         declare -A METRIC_VALS

#         for SEED in "${SEEDS[@]}"; do
#             echo "    -- Seed ${SEED} --"

#             SEED_OUT=$(python main.py \
#                 --config         "$CONFIG" \
#                 --detection      True \
#                 --category       "$CATEGORY" \
#                 --load_chp       "$LOAD_CHP" \
#                 --checkpoint_dir "$BASE_PATH" \
#                 --seed           "$SEED" \
#                 2>&1 | tee -a "$FULL_LOG")

#             echo "    Done seed ${SEED}" | tee -a "$FULL_LOG"

#             while IFS= read -r line; do
#                 if [[ "$line" == *"Per-image Pixel AUROC (mean):"* ]]; then
#                     val=$(echo "$line" | awk -F': ' '{print $2}' | tr -d ' ')
#                     METRIC_VALS[pixel_auroc]="${METRIC_VALS[pixel_auroc]} $val"
#                 elif [[ "$line" =~ ^"SNR:" ]]; then
#                     val=$(echo "$line" | awk '{print $2}')
#                     METRIC_VALS[snr]="${METRIC_VALS[snr]} $val"
#                 fi
#             done <<< "$SEED_OUT"
#         done

#         for metric in pixel_auroc snr; do
#             vals_arr=(${METRIC_VALS[$metric]})
#             actual_n=${#vals_arr[@]}
#             if [ "$actual_n" -eq 0 ]; then continue; fi
#             result=$(stats "${vals_arr[@]}")
#             mean_v=$(echo "$result" | awk '{print $1}')
#             std_v=$(echo  "$result" | awk '{print $2}')
#             line="  $(printf '%-15s' "$metric")  mean=${mean_v}  std=${std_v}  (n=${actual_n})"
#             echo "$line" | tee -a "$AVG_FILE"
#         done

#         unset METRIC_VALS
#         declare -A METRIC_VALS
#         echo "  ---" >> "$AVG_FILE"
#     done
# done

# echo ""
# echo "All done."
# echo "  Full log : $FULL_LOG"
# echo "  Averages : $AVG_FILE"



#!/bin/bash
# run_all_categories.sh

CONFIG="config.yaml"

DDAD_CKPT_BASE="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec"
MVTEC_CKPT_BASE="/data/akherandish3/MVTec"

ALL_SEEDS=(42 123 456 789 1337 2024 99 7 314 2718 555 888 1000 2000 3000 4000 5000 6000 7000 8000)
SEED_COUNTS=(1 10 20)

CATEGORIES=(
    # "CT:2000:${DDAD_CKPT_BASE}/combined"
    "faces:2000:${DDAD_CKPT_BASE}/faces"
    "bottle:1000:${MVTEC_CKPT_BASE}"
    "cable:3000:${MVTEC_CKPT_BASE}"
    "capsule:1500:${MVTEC_CKPT_BASE}"
    "carpet:2500:${MVTEC_CKPT_BASE}"
    "grid:2000:${MVTEC_CKPT_BASE}"
    "hazelnut:2000:${MVTEC_CKPT_BASE}"
    "leather:2000:${MVTEC_CKPT_BASE}"
    "metal_nut:3000:${MVTEC_CKPT_BASE}"
    "pill:1000:${MVTEC_CKPT_BASE}"
    "screw:2000:${MVTEC_CKPT_BASE}"
    "tile:1000:${MVTEC_CKPT_BASE}"
    "toothbrush:2000:${MVTEC_CKPT_BASE}"
    "transistor:2000:${MVTEC_CKPT_BASE}"
    "wood:2000:${MVTEC_CKPT_BASE}"
    "zipper:1000:${MVTEC_CKPT_BASE}"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_LOG="results_full_${TIMESTAMP}.txt"
AVG_FILE="results_averaged_${TIMESTAMP}.txt"

# ── Redirect ALL stdout+stderr to FULL_LOG while keeping terminal output ──────
exec > >(tee -a "$FULL_LOG") 2>&1
# From this point on, EVERYTHING (python output, warnings, errors) goes to both
# the terminal and FULL_LOG automatically — no need for tee in individual calls
# ─────────────────────────────────────────────────────────────────────────────

echo "DDAD Detection — Full Log"
echo "Timestamp: $TIMESTAMP"
echo "Seeds: ${ALL_SEEDS[*]}"

py_stats() {
    python3 -c "
import sys, math
vals = list(map(float, sys.argv[1:]))
n = len(vals)
m = sum(vals)/n
s = math.sqrt(sum((x-m)**2 for x in vals)/(n-1)) if n>1 else 0.0
print(f'{m:.4f} {s:.4f}')
" "$@"
}

# Start averages file
echo "DDAD Detection — Averaged Results" >  "$AVG_FILE"
echo "Timestamp: $TIMESTAMP"             >> "$AVG_FILE"
echo "Seeds used: ${ALL_SEEDS[*]}"       >> "$AVG_FILE"

for ENTRY in "${CATEGORIES[@]}"; do
    CATEGORY=$(echo "$ENTRY" | cut -d: -f1)
    LOAD_CHP=$(echo  "$ENTRY" | cut -d: -f2)
    BASE_PATH=$(echo "$ENTRY" | cut -d: -f3-)
    CKPT_PATH="${BASE_PATH}/${CATEGORY}/${LOAD_CHP}"

    echo ""
    echo "########################################"
    echo "# Category : ${CATEGORY}"
    echo "# Checkpoint: ${CKPT_PATH}"
    echo "########################################"

    if [ ! -e "$CKPT_PATH" ]; then
        echo "[SKIP] Checkpoint not found: ${CKPT_PATH}"
        echo "" >> "$AVG_FILE"
        echo "[SKIP] ${CATEGORY} — checkpoint not found: ${CKPT_PATH}" >> "$AVG_FILE"
        continue
    fi

    for N in "${SEED_COUNTS[@]}"; do
        SEEDS=("${ALL_SEEDS[@]:0:$N}")
        AUROC_VALS=()
        SNR_VALS=()

        echo ""
        echo "  === N=${N} seeds: ${SEEDS[*]} ==="

        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "  ---- Seed ${SEED} ----"

            TMPOUT=$(mktemp)

            # Run — output goes to terminal + FULL_LOG automatically via exec above
            python main.py \
                --config         "$CONFIG" \
                --detection      True \
                --category       "$CATEGORY" \
                --load_chp       "$LOAD_CHP" \
                --checkpoint_dir "$BASE_PATH" \
                --seed           "$SEED" \
                2>&1 | tee "$TMPOUT"
            # tee "$TMPOUT" saves a copy for parsing; exec above handles FULL_LOG

            AUROC=$(grep "Per-image Pixel AUROC (mean):" "$TMPOUT" \
                    | tail -1 | awk -F': ' '{print $2}' | tr -d '[:space:]')
            SNR=$(  grep "^SNR:"                          "$TMPOUT" \
                    | tail -1 | awk '{print $2}'          | tr -d '[:space:]')
            rm -f "$TMPOUT"

            if [ -n "$AUROC" ]; then
                AUROC_VALS+=("$AUROC")
                echo "  [parsed] Pixel AUROC=${AUROC}  SNR=${SNR}"
            else
                echo "  [WARN] Could not parse metrics for seed ${SEED}"
            fi
            [ -n "$SNR" ] && SNR_VALS+=("$SNR")

            echo "  ---- Done seed ${SEED} ----"
        done

        # Write averages
        {
            echo ""
            echo "Category: ${CATEGORY} | Checkpoint: ${LOAD_CHP} | N_seeds: ${N}"
            if [ ${#AUROC_VALS[@]} -gt 0 ]; then
                read AUROC_MEAN AUROC_STD <<< $(py_stats "${AUROC_VALS[@]}")
                echo "  Pixel AUROC  mean=${AUROC_MEAN}  std=${AUROC_STD}  (n=${#AUROC_VALS[@]})"
                echo "  Raw values : ${AUROC_VALS[*]}"
            else
                echo "  Pixel AUROC  [no data collected]"
            fi
            if [ ${#SNR_VALS[@]} -gt 0 ]; then
                read SNR_MEAN SNR_STD <<< $(py_stats "${SNR_VALS[@]}")
                echo "  SNR          mean=${SNR_MEAN}  std=${SNR_STD}  (n=${#SNR_VALS[@]})"
                echo "  Raw values : ${SNR_VALS[*]}"
            else
                echo "  SNR          [no data collected]"
            fi
            echo "  ---"
        } >> "$AVG_FILE"
    done
done

echo ""
echo "========================================"
echo "All done."
echo "  Full output : $FULL_LOG"
echo "  Averages    : $AVG_FILE"
echo "========================================"