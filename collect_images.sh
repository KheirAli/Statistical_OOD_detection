SAMPLE_NAME="samples_98"
SRC_BASE="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME"
DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/recon"

mkdir -p "$DST"

count=0

for i in {8..23}; do
    SRC="$SRC_BASE/Combined_0.5_sigma_batched_${i}_4/inpainting/recon"

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

echo "Copied $count images."

LABEL_SRC="/data/akherandish3/Statistical_OOD_detection/faces_guided/$SAMPLE_NAME/Combined_0.5_sigma_batched_8_4/inpainting/label/0_00000.png"

LABEL_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/label"
INPUT_DST="/data/akherandish3/Statistical_OOD_detection/results_patches_ddad_native_faces_random_1/$SAMPLE_NAME/ddad_native_0_4/inpainting/input"

mkdir -p "$LABEL_DST" "$INPUT_DST"

cp "$LABEL_SRC" "$LABEL_DST/0_00000.png"
cp "$LABEL_SRC" "$INPUT_DST/0_00000.png"