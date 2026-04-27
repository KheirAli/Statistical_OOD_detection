#!/bin/bash

MVTEC_CKPT_BASE="/data/akherandish3/MVTec"
MVTEC_DATA_BASE="/data/akheirandish3/mvtec_ad"

# Loop over each category folder in the MVTec checkpoints directory
for cat_dir in "$MVTEC_CKPT_BASE"/*/; do
    cat=$(basename "$cat_dir")
    
    echo "========================================"
    echo "Processing category: $cat"
    echo "========================================"
    
    # Find the numeric checkpoint file (ignoring 'feat...' folders)
    # This filters for file/folder names that start with a number
    ckpt_dir=$(find "$cat_dir" -maxdepth 1 -mindepth 1 -name "[0-9]*" | head -n 1)
    
    if [ -z "$ckpt_dir" ]; then
        echo "No numeric checkpoint found for $cat. Skipping..."
        continue
    fi
    
    echo "Using checkpoint: $ckpt_dir"

    # Define the path to the test images for this category
    test_base="$MVTEC_DATA_BASE/$cat/test"
    if [ ! -d "$test_base" ]; then
        echo "Test directory $test_base not found. Skipping..."
        continue
    fi
    
    # Loop over all subdirectories (defect types + 'good') inside the test folder
    for sub_dir in "$test_base"/*/; do
        # Extract the defect name (e.g., 'bent_wire', 'good')
        subcat=$(basename "$sub_dir")
        
        echo "  -> Running subcategory: $subcat"
        
        out_root="./results_patches_ddad_native_${cat}_${subcat}_1"
        
        # Strip the trailing slash from sub_dir for the argument
        image_dir="${sub_dir%/}"
        
        CUDA_VISIBLE_DEVICES=5 python tools/generate_recons.py \
            --recon_config configs/recon/ddad_native_cable.yaml \
            --ckpt "$ckpt_dir" \
            --image_dir "$image_dir" \
            --out_root "$out_root"
    done
done
EOF