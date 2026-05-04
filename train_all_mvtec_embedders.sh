#!/bin/bash

# Base directory for MVTec dataset
MVTEC_DATA_BASE="/data/akheirandish3/mvtec_ad"

# Directory to save trained embedders
# We save them in a dedicated folder with the category name in the filename
# so that evaluation scripts can easily load them string formatting: f"models/mvtec_embedders/pixel_autoencoder_{category}.pth"
SAVE_DIR="models/mvtec_embedders"
mkdir -p "$SAVE_DIR"

# Loop over each category folder in the MVTec directory
for cat_dir in "$MVTEC_DATA_BASE"/*/; do
    # Extract the category name (e.g., 'cable', 'capsule', 'bottle')
    cat=$(basename "$cat_dir")
    
    echo "========================================"
    echo "Training embedder for category: $cat"
    echo "========================================"
    
    # MVTec normal samples are inside 'train/good'
    train_dir="$cat_dir/train/good"
    
    if [ ! -d "$train_dir" ]; then
        echo "Training directory $train_dir not found. Skipping $cat..."
        continue
    fi
    
    save_path="$SAVE_DIR/pixel_autoencoder_${cat}.pth"
    
    # Run the training script
    # Note: Adjust CUDA_VISIBLE_DEVICES if you want to run on a specific GPU
    CUDA_VISIBLE_DEVICES=0 python train_embedder.py \
        --data_dir "$train_dir" \
        --save_model "$save_path" \
        --num_epochs 30 \
        --latent_dim 3 \
        --resnet_name resnet101
        
    echo "Finished training for $cat."
    echo "Model saved to $save_path"
    echo ""
done

echo "All training jobs completed! Models are saved in $SAVE_DIR"
