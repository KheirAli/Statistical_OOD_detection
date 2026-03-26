# #!/bin/bash

# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"

# # Define all 16 patches with GPU assignments
# patches=(
#     "0 0 1 64 0 64 6"      # row col top bottom left right gpu
#     "1 1 1 64 64 128 4"
#     "2 2 1 64 128 192 3"
#     "3 3 1 64 192 256 5"
#     "4 0 1 128 0 64 5"
#     "5 1 1 128 64 128 5"
#     "6 2 1 128 128 192 5"
#     "7 3 1 128 192 256 2"
#     "8 0 1 192 0 64 1"
#     "9 1 1 192 64 128 1"
#     "10 2 1 192 128 192 2"
#     "11 3 1 192 192 256 2"
#     "12 0 1 256 0 64 1"
#     "13 1 1 256 64 128 1"
#     "14 2 1 256 128 192 7"
#     "15 3 1 256 192 256 7"
# )
# for num_samples in $(seq 0 1); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_5" \
#             --seed $(((num_samples - 1) * 15 + row)) \
#             --box_coords $top $bottom $left $right &

#     done
#     wait
    
# done
# #     CUDA_VISIBLE_DEVICES=7 python3 sample_condition.py \
# #         --model_config=configs/model_config.yaml \
# #         --diffusion_config=configs/diffusion_config.yaml \
# #         --task_config=configs/inpainting_config.yaml \
# #         --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_2" \
# #         --seed ${num_samples} \
# #         --box_coords 1 1 0 1 &

# echo "Done!"

############################# HERE STARTS NEW CODE #############################
# python3 super_pixel_generation.py \
#     --input_image=/home/akheirandish3/diffusion-posterior-sampling/data/samples/00092_1.png \
#     --n_segments=150
# wait

# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"

# # Define all 16 patches with GPU assignments
# patches=(
#     "0 0 0 64 0 64 2"      # row col top bottom left right gpu
#     "0 1 1 64 64 128 2"
#     "0 2 2 64 128 192 2"
#     "0 3 3 64 192 256 2"
#     "1 0 4 128 0 64 1"
#     "1 1 5 128 64 128 1"
#     "1 2 6 128 128 192 1"
#     "1 3 7 128 192 256 1"
#     "2 0 8 192 0 64 1"
#     "2 1 9 192 64 128 1"
#     "2 2 10 192 128 192 5"
#     "2 3 11 192 192 256 5"
#     "3 0 12 256 0 64 1"
#     "3 1 13 256 64 128 1"
#     "3 2 14 256 128 192 7"
#     "3 3 15 256 192 256 7"
#     "4 0 16 320 0 64 6"
#     "4 1 17 320 64 128 6"
#     "4 2 18 320 128 192 5"
#     "4 3 19 320 192 256 5"
#     "5 0 20 384 0 64 2"
#     "5 1 21 384 64 128 2"
#     "5 2 22 384 128 192 7"
#     "5 3 23 384 192 256 7"
#     "4 1 24 320 64 128 4"
#     "4 2 25 320 128 192 7"
#     "4 3 26 320 192 256 7"
#     "5 0 27 384 0 64 6"
#     "5 1 28 384 64 128 6"
#     "5 2 29 384 128 192 3"
#     "5 3 30 384 192 256 3"
#     "5 0 31 384 0 64 3"
#     "5 1 32 384 64 128 3"
#     "5 2 33 384 128 192 3"
#     "5 3 34 384 192 256 3"
# )
# for num_samples in $(seq 1 2); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

# patches=(
#     "0 0 35 64 0 64 2"      # row col top bottom left right gpu
#     "0 1 36 64 64 128 2"
#     "0 2 37 64 128 192 2"
#     "0 3 38 64 192 256 2"
#     "1 0 39 128 0 64 1"
#     "1 1 40 128 64 128 1"
#     "1 2 41 128 128 192 1"
#     "1 3 42 128 192 256 1"
#     "2 0 43 192 0 64 5"
#     "2 1 44 192 64 128 5"
#     "2 2 45 192 128 192 5"
#     "2 3 46 192 192 256 5"
#     "3 0 47 256 0 64 1"
#     "3 1 48 256 64 128 1"
#     "3 2 49 256 128 192 7"
#     "3 3 50 256 192 256 7"
#     "4 0 51 320 0 64 6"
#     "4 1 52 320 64 128 6"
#     "4 2 53 320 128 192 5"
#     "4 3 54 320 192 256 5"
#     "5 0 55 384 0 64 2"
#     "5 1 56 384 64 128 2"
#     "5 2 57 384 128 192 7"
#     "5 3 58 384 192 256 7"
#     "4 1 59 320 64 128 4"
#     "4 2 60 320 128 192 7"
#     "4 3 61 320 192 256 7"
#     "5 0 62 384 0 64 6"s
#     "5 1 63 384 64 128 6"
#     "5 2 64 384 128 192 3"
#     "5 3 65 384 192 256 3"
#     "5 0 66 384 0 64 3"
#     "5 1 67 384 64 128 3"
#     "5 2 68 384 128 192 3"s
#     "5 3 69 384 192 256 3"
# )
# for num_samples in $(seq 1 2); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/superpixel_scar_${top}" \
#             --seed ${num_samples} \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png &

#     done
#     wait
# done

# #     --mask_prob 2.0 \
# #     --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png 
# wait
# python3 step_1.py \
#   --global_label_path "/home/akheirandish3/diffusion-posterior-sampling/data/samples/00092_1.png" \
#   --mask_path "/home/akheirandish3/diffusion-posterior-sampling/data/mask.png" \
#   --output_path "/home/akheirandish3/diffusion-posterior-sampling/data/uncertanity_map_1.png" \
#   --n_superpixels 69

# wait

# patches=(
#     "0 0 1 64 0 64 6"      # row col top bottom left right gpu
#     "1 1 1 64 64 128 4"
#     "2 2 1 64 128 192 3"
#     "3 3 1 64 192 256 5"
#     "4 0 1 128 0 64 5"
#     "5 1 1 128 64 128 5"
#     "6 2 1 128 128 192 5"
#     "7 3 1 128 192 256 2"
#     "8 0 1 192 0 64 1"
#     "9 1 1 192 64 128 1"
#     "10 2 1 192 128 192 2"
#     "11 3 1 192 192 256 2"
#     "12 0 1 256 0 64 1"
#     "13 1 1 256 64 128 1"
#     "14 2 1 256 128 192 7"
#     "15 3 1 256 192 256 7"
# )
# for num_samples in $(seq 0 1); do
#     for patch in "${patches[@]}"; do
#         read -r row col top bottom left right gpu <<< "$patch"
        
#         CUDA_VISIBLE_DEVICES=$gpu python3 sample_condition.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/baseline_scar/new_label_5" \
#             --seed $(((num_samples - 1) * 15 + row)) \
#             --box_coords $top $bottom $left $right \
#             --mask_prob 2.0 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/uncertanity_map_1.png &
#     done
#     wait
    
# done
# IMAGE_SIZE=256
# PATCH_SIZE=64
# BASE_SAVE_DIR="./results_patches"
# # SAMPLES_DIR="./data/samples"
# SAMPLES_DIR="/data/akheirandish3/mvtec_ad/test_image"
# SAMPLES_NEXT_DIR="./data/samples_next"

# # List of sample files to process
# SAMPLE_FILES=(
#     # "00092_1.png"
#     "00105_1.png"
#     # "00243.png"
#     # # "00107.png"
#     # "00015_out_pnegg.png"
# )
# # echo "Done!"
# CUDA_VISIBLE_DEVICES=5 python3 sample_batch.py \
#             --model_config=configs/model_config.yaml \
#             --diffusion_config=configs/diffusion_config.yaml \
#             --task_config=configs/inpainting_config.yaml \
#             --save_dir="${BASE_SAVE_DIR}/img_blur" \
#             --data_root="${SAMPLES_DIR}" \
#             --seed 0 \
#             --box_coords 80 4 0 0 \
#             --mask_prob 0.9 \
#             --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png\
#             --num_measurements 4 &

    
IMAGE_SIZE=256
PATCH_SIZE=64
BASE_SAVE_DIR="./results_patches"
SAMPLES_DIR="/data/akheirandish3/mvtec_ad/test_image"
# SAMPLES_NEXT_DIR="/data/akheirandish3/mvtec_ad/cable/test/combined"
SAMPLES_NEXT_DIR="/data2/akheirandish3/id_new_warped_images/OOD_moderate_medium/1"
# List of sample files to process
# SAMPLE_FILES=(
#     "000.png"
#     "001.png"
#     "002.png"
#     "003.png"
#     "004.png"
#     "005.png"
#     "006.png"
#     "007.png"
#     "008.png"
#     "009.png"
#     "010.png"
#     # "011.png"
# )
SAMPLE_FILES=(
    "image_0000.png"
    "image_0001.png"
    "image_0002.png"
    "image_0003.png"
    "image_0004.png"
    "image_0005.png"
    "image_0006.png"
    "image_0007.png"
    "image_0008.png"
    "image_0009.png"
    "image_0010.png"
    # "011.png"
)

get_patches() {
    local bottom=$1
    
    local patches=(
        # "0 0 0 $bottom 0 64 3"
        # "0 1 1 $bottom 64 128 3"
        # "0 2 2 $bottom 128 192 3"
        "0 3 3 $bottom 192 256 0"
        # "1 0 4 $bottom 0 64 2"
        "1 1 5 $bottom 64 128 2"
        "1 2 6 $bottom 128 192 2"
        "1 3 7 $bottom 192 256 1"
        "2 0 8 $bottom 0 64 1"
        "2 1 9 $bottom 64 128 1"
        # "2 2 10 $bottom 128 192 1"
        # "2 3 11 $bottom 192 256 6"
        # "3 0 12 $bottom 0 64 6"
        # "3 1 13 $bottom 64 128 6"
        # "3 2 14 $bottom 128 192 6"
        # "3 3 15 $bottom 192 256 7"
        # "4 0 16 $bottom 0 64 4"
        # "4 1 17 $bottom 64 128 4"
        # "4 2 18 $bottom 128 192 4"
        # "4 3 19 $bottom 192 256 2"
        # "5 0 20 $bottom 0 64 5"
        # "5 1 21 $bottom 64 128 5"
        # "5 2 22 $bottom 128 192 5"
        # "5 3 23 $bottom 192 256 7"
        # "4 1 24 $bottom 64 128 7"
        # "4 2 25 $bottom 128 192 7"
        # "4 3 26 $bottom 192 256 7"
        # "5 0 27 $bottom 0 64 7"
    )
    
    printf '%s\n' "${patches[@]}"
}

process_patches_batch() {
    local bottom=$1
    local sample_name=$2
    local num_samples=$3
    echo "Processing patches with bottom=$bottom for sample=$sample_name"
    
    while IFS= read -r patch; do
        read -r row col top _ left right gpu <<< "$patch"
        
        CUDA_VISIBLE_DEVICES=$gpu python3 sample_batch.py \
            --model_config=configs/model_config.yaml \
            --diffusion_config=configs/diffusion_config.yaml \
            --task_config=configs/inpainting_config.yaml \
            --save_dir="${BASE_SAVE_DIR}/samples_${sample_name}/Combined_one_sigma_batched_${top}_${bottom}" \
            --data_root="${SAMPLES_DIR}" \
            --seed ${num_samples} \
            --box_coords $top $bottom $left $right \
            --mask_prob 0.9 \
            --mask_path /home/akheirandish3/diffusion-posterior-sampling/data/mask.png \
            --num_measurements 4 &
    done < <(get_patches "$bottom")
    
    wait
}

process_sample_batch() {
    local filename=$1
    local sample_name="${filename%.*}"
    
    echo "=========================================="
    echo "Processing sample: $filename"
    echo "=========================================="
    # Move file from samples_next to samples with new naming
    if [ -f "${SAMPLES_NEXT_DIR}/${filename}" ]; then
        ln -sf "${SAMPLES_NEXT_DIR}/${filename}" "${SAMPLES_DIR}/${filename}"
        echo "✓ Linked: ${SAMPLES_NEXT_DIR}/${filename} -> ${SAMPLES_DIR}/${filename}"
    else
        echo "✗ Warning: ${SAMPLES_NEXT_DIR}/${filename} not found!"
        return 1
    fi
    # Generate superpixels for this sample
    echo "Generating superpixels..."
    python3 super_pixel_generation.py \
        --input_image="${SAMPLES_DIR}/${sample_name}.png"
    
    wait
    # Process patches for different bottom values (0, 1, 2)
    for bottom in 4; do
        process_patches_batch "$bottom" "$sample_name" "3"
    done
    echo "✓ Completed sample: $filename"
    echo "Moving file back to samples_next..."
    if [ -f "${SAMPLES_DIR}/${sample_name}.png" ]; then
        rm -f "${SAMPLES_DIR}/${sample_name}.png"
        echo "✓ Removed symlink: ${SAMPLES_DIR}/${sample_name}.png"
    else
        echo "✗ Warning: Could not find ${SAMPLES_DIR}/${filename} to move back!"
    fi
}
# Main execution
echo "Starting batch processing..."

# for sample_file in "${SAMPLE_FILES[@]}"; do
#     process_sample "$sample_file"
# done
for sample_file in "${SAMPLE_FILES[@]}"; do
    process_sample_batch "$sample_file"
    wait
done
echo "=========================================="
echo "All samples processed successfully!"
echo "=========================================="