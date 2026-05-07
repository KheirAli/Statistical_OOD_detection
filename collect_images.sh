# SAMPLE_NAME="samples_98"
# SRC_BASE="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME"
# DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"

# SAMPLE_FILES=( # anomaly
#     "P00209_OL.png"
#     "P01115_OL.png"
#     "P01207_OL.png"
#     "P01733_OL.png"
#     "P01818_OL.png"
#     "P01861_OL.png"
#     "P01927_OL.png"
#     "P01950_OL.png"
#     "P01965_OL.png"
#     "P02052_OL.png"
#     "P02117_OL.png"
#     "P02244_OL.png"
#     "P02254_OL.png"
#     "P02330_OL.png"
#     "P03237_OL.png"
#     "P03262_OL.png"
#     "P03921_OL.png"
#     "P04460_OL.png"
#     "P04469_OL.png"
# )

SAMPLE_NAMES=(
    "samples_92"
    "samples_11"
    "samples_12"
    "samples_13"
    "samples_14"
    "samples_15"
    "samples_16"
    "samples_17"
    # "samples_00105_1.png"
    "samples_107"
    # "00015_out_pnegg.png"
    "samples_129"
    "samples_98"
    "samples_80"
    "samples_40"
    "samples_44"
    "samples_61"
    "samples_49"
    "samples_65"
    "samples_117"
    "samples_2"
    "samples_1"
    "samples_3"
    "samples_4"
    "samples_5"
    "samples_6"
    "samples_7"
    "samples_8"
    "samples_9"
    "samples_10"
    "samples_120"
    "samples_59"
    "samples_86"
)

# SAMPLE_NAMES=(
#     "samples_P01809_OL"
#     "samples_P01810_OL"
#     "samples_P01827_OL"
#     "samples_P01874_OL"
#     "samples_P01942_OL"
#     "samples_P01951_OL"
#     "samples_P01973_OL"
#     "samples_P02003_OL"
#     "samples_P02013_OL"
#     "samples_P02028_OL"
#     "samples_P02079_OL"
#     "samples_P02082_OL"
#     "samples_P02095_OL"
#     "samples_P02115_OL"
#     "samples_P02148_OL"
#     "samples_P02170_OL"
#     "samples_P02195_OL"
#     "samples_P02220_OL"
#     "samples_P02229_OL"
#     "samples_P02238_OL"
#     "samples_P02253_OL"
#     "samples_P03265_OL"
#     "samples_P03288_OL"
#     "samples_P03672_OL"
#     "samples_P03685_OL"
#     "samples_P03695_OL"
#     "samples_P04649_OL"
#     "samples_P04666_OL"
#     "samples_P04789_OL"
#     "samples_P04803_OL"
#     "samples_P04809_OL"
#     "samples_P04872_OL"
# )

for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
    # SRC_BASE="/data/akherandish3/Statistical_OOD_detection/Xray_guided/$SAMPLE_NAME"
    SRC_BASE="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME"
    DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"
    # DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_scissors_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"
    mkdir -p "$DST"

    count=0

    for i in {8..23}; do
        SRC="$SRC_BASE/Combined_1.0_sigma_batched_${i}_4/inpainting/recon"
        if [ -d "$SRC" ]; then
            echo "Copying from $SRC"
            for img in $(find "$SRC" -maxdepth 1 -type f | sort); do
                newname=$(printf "%d_0_00000.png" "$count")
                cp "$img" "$DST/$newname"
                echo "$img -> $DST/$newname"
                count=$((count + 1))
            done
        else
            echo "Missing directory: $SRC"
        fi
    done

    echo "Copied $count images for $SAMPLE_NAME."

    # LABEL_SRC="/data/akherandish3/Statistical_OOD_detection/Xray_guided/$SAMPLE_NAME/Combined_1.0_sigma_batched_16_4/inpainting/label/0_00000.png"
    LABEL_SRC="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME/Combined_1.0_sigma_batched_16_4/inpainting/label/0_00000.png"

    # LABEL_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_scissors_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/label"
    # INPUT_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_scissors_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/input"

    LABEL_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/label"
    INPUT_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_2/$SAMPLE_NAME/ddad_native_0_4/inpainting/input"
    mkdir -p "$LABEL_DST" "$INPUT_DST"

    cp "$LABEL_SRC" "$LABEL_DST/0_00000.png"
    cp "$LABEL_SRC" "$INPUT_DST/0_00000.png"
done


# SAMPLE_NAME="samples_P00209_OL"
# SRC_BASE="/data/akherandish3/Statistical_OOD_detection/Xray_guided/$SAMPLE_NAME"
# DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_anomaly_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"
# # DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_scissors_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"
# mkdir -p "$DST"

# count=0

# for i in {8..23}; do
#     SRC="$SRC_BASE/Combined_1.0_sigma_batched_${i}_4/inpainting/recon" #combined_5

#     if [ -d "$SRC" ]; then
#         echo "Copying from $SRC"

#         for img in $(find "$SRC" -maxdepth 1 -type f | sort); do
#             newname=$(printf "%d_0_00000.png" "$count")
#             cp "$img" "$DST/$newname"
#             echo "$img -> $DST/$newname"
#             count=$((count + 1))
#         done
#     else
#         echo "Missing directory: $SRC"
#     fi
# done

# echo "Copied $count images."

# LABEL_SRC="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME/Combined_0.5_sigma_batched_8_4/inpainting/label/0_00000.png"

# LABEL_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/label"
# INPUT_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/input"

# LABEL_SRC="/data/akherandish3/Statistical_OOD_detection/Xray_guided/$SAMPLE_NAME/Combined_0.8_sigma_batched_8_4/inpainting/label/0_00000.png"
# LABEL_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_anomaly_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/label"
# INPUT_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_xray_anomaly_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/input"
# mkdir -p "$LABEL_DST" "$INPUT_DST"

# cp "$LABEL_SRC" "$LABEL_DST/0_00000.png"
# cp "$LABEL_SRC" "$INPUT_DST/0_00000.png"