# /data/akheirandish3/mvtec_ad/faces/extras
# CUDA_VISIBLE_DEVICES=1 python train_embedder.py --data_dir /data/akheirandish3/mvtec_ad/cable/train/good --save_model models/pixel_autoencoder_cable_new.pth
# CUDA_VISIBLE_DEVICES=1 python train_embedder.py --data_dir /data2/akheirandish3/id_new_warped_images/ID_same_base --save_model models/pixel_autoencoder_CT_same_data.pth --num_epochs 20 --latent_dim 3
# /data2/akheirandish3/id_new_warped_images/ID_same_base
# img_t: torch [1,3,H,W]
import gc
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

# (optional) your DDAD-style local context; keeps (Hi,Wi) but mixes 3x3 neighborhoods
def patchify_context(features: torch.Tensor, patchsize=3, stride=1):
    # features: [B,C,H,W] -> unfold -> average over patch -> [B,C,H,W]
    padding = (patchsize - 1) // 2
    unfolder = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=padding)
    B, C, H, W = features.shape
    unfolded = unfolder(features)                      # [B, C*ps*ps, H*W]
    unfolded = unfolded.view(B, C, patchsize*patchsize, H*W)
    pooled = unfolded.mean(dim=2)                      # [B, C, H*W]
    return pooled.view(B, C, H, W)

class ResNetPixelEmbedder(nn.Module):
    """
    Returns full-resolution pixel embedding [B, C_embed, H, W]
    by concatenating upsampled intermediate resnet features.
    """
    def __init__(self, resnet_name="resnet18", layers=("layer1","layer2","layer3"),
                 out_size=None, use_imagenet_norm=True, use_patch_context=True,
                 proj_dim_per_layer=None):
        super().__init__()

        # backbone
        if resnet_name == "resnet18":
            # Fallback for older torchvision versions
            try:
                net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except AttributeError:
                net = models.resnet18(pretrained=True)
        elif resnet_name == "resnet50":
            try:
                net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet50(pretrained=True)
        elif resnet_name == "resnet101":    # <--- ADD THIS BLOCK
            try:
                net = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            except AttributeError:
                net = models.resnet101(pretrained=True)
        else:
            raise ValueError("resnet_name must be resnet18, resnet50, or resnet101")
        net.eval()

        return_nodes = {ln: ln for ln in layers}
        self.extractor = create_feature_extractor(net, return_nodes=return_nodes)
        self.layers = layers
        self.out_size = out_size  # if None, uses input H,W
        self.use_patch_context = use_patch_context

        # optional 1x1 projections to keep C manageable
        self.proj = nn.ModuleDict()
        self.proj_dim_per_layer = proj_dim_per_layer
        if proj_dim_per_layer is not None:
            # need to know channels; easiest: infer from resnet type
            # resnet18: layer1=64, layer2=128, layer3=256, layer4=512
            # resnet50: layer1=256, layer2=512, layer3=1024, layer4=2048
            ch_map = {}
            if resnet_name == "resnet18":
                ch_map = {"layer1":64, "layer2":128, "layer3":256, "layer4":512}
            else:
                ch_map = {"layer1":256, "layer2":512, "layer3":1024, "layer4":2048}

            for ln in layers:
                self.proj[ln] = nn.Conv2d(ch_map[ln], proj_dim_per_layer, kernel_size=1, bias=False)

        self.use_imagenet_norm = use_imagenet_norm
        if use_imagenet_norm:
            self.register_buffer("mean", torch.tensor([0.485,0.456,0.406])[None,:,None,None])
            self.register_buffer("std",  torch.tensor([0.229,0.224,0.225])[None,:,None,None])

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        x: [B,3,H,W] float, in [0,1] or [-1,1]
        returns: emb [B,C_embed,H,W]
        """
        assert x.ndim == 4 and x.shape[1] == 3
        B, _, H, W = x.shape

        # map to [0,1] if needed
        if x.min() < 0:
            x = (x + 1) / 2.0

        if self.use_imagenet_norm:
            x = (x - self.mean) / self.std

        feats = self.extractor(x)  # dict layer -> tensor [B,Ci,Hi,Wi]

        outH, outW = (self.out_size, self.out_size) if self.out_size is not None else (H, W)

        ups = []
        for ln in self.layers:
            f = feats[ln]
            if self.use_patch_context:
                f = patchify_context(f, patchsize=3, stride=1)
            if self.proj_dim_per_layer is not None:
                f = self.proj[ln](f)
            f = F.interpolate(f, size=(outH, outW), mode="bilinear", align_corners=False)
            f = F.normalize(f, dim=1)  # stabilize scale across channels
            ups.append(f)

        emb = torch.cat(ups, dim=1)  # [B, sumCi, outH, outW]
        return emb
    


import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
        self.image_paths = sorted(
            str(path)
            for path in Path(root_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class PixelAutoEncoder(nn.Module):
    def __init__(self, input_dim=1417, latent_dim=5):
        super().__init__()
        # 1x1 Convs with BatchNorm to prevent dead ReLUs
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=1),
            nn.Tanh()  # <--- Force z strictly between [-1, 1]

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, input_dim, kernel_size=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # ResNet outputs are L2 normalized, so we enforce L2 norm on reconstruction
        x_hat = F.normalize(x_hat, dim=1)
        return x_hat, z

import argparse
parser = argparse.ArgumentParser(description="Train Pixel AutoEncoder")
parser.add_argument("--data_dir", type=str, default="/data/akheirandish3/mvtec_ad/faces/extras", help="Path to training images")
parser.add_argument("--save_model", type=str, default="models/pixel_autoencoder_faces.pth", help="Path to save the trained model")
parser.add_argument("--resnet_name", type=str, default="resnet101", choices=["resnet18","resnet50","resnet101"], help="ResNet backbone for embedding")
parser.add_argument("--latent_dim", type=int, default=5, help="Dimensionality of the latent space in the autoencoder")
parser.add_argument("--proj_dim_per_layer", type=int, default=None, help="If set, applies a 1x1 conv to reduce each ResNet layer to this many channels before concatenation")
parser.add_argument("--use_patch_context", action="store_true", help="If set, applies patchify_context to ResNet features for local context")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
# parser.add_argument("--latent_dim", type=int, default=5, help="Dimensionality of the latent space in the autoencoder")
args = parser.parse_args()

# Setup data loading
data_dir = args.data_dir
# Resize could be added if images vary, using 256 here for example
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = ImageDataset(data_dir, transform=transform)

# Split into 90% train, 10% val
val_size = max(1, int(0.1 * len(dataset)))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

embedder = ResNetPixelEmbedder(
    resnet_name="resnet101",
    layers=("layer1","layer2","layer3"),
    out_size=None,
    use_patch_context=True,
    proj_dim_per_layer=None
).to("cuda").eval()

# Determine embedder output dimension dynamically
dummy_input = torch.zeros(1, 3, 256, 256).to("cuda")
with torch.no_grad():
    embed_dim = embedder(dummy_input).shape[1]

model = PixelAutoEncoder(input_dim=embed_dim, latent_dim=args.latent_dim).to("cuda")
# For normalized vectors, Cosine Similarity is usually better than MSE. 
# We'll use 1 - cosine_similarity as the distance.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = args.num_epochs
best_val_loss = float('inf')

save_dir = os.path.dirname(args.save_model)
if save_dir:
    os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, images in enumerate(train_loader):
        images = images.to("cuda") # [B, 3, H, W]
        
        with torch.no_grad():
            res_feats = embedder(images) # [B, C_res, H, W]
            # using only the embeddings
            combined_feats = res_feats

        x_hat, z = model(combined_feats)
        
        # Loss: 1.0 - mean(cosine_similarity)
        # Cosine similarity is 1.0 for perfect match, so 1.0 - 1.0 = 0 loss.
        loss = 1.0 - F.cosine_similarity(x_hat, combined_feats, dim=1).mean()

        # mse_loss = F.mse_loss(x_hat, combined_feats)
        
        # # 2. Divergence (Variance Spread) Loss
        # # Calculate variance of z across Batch, H, and W dimensions
        # z_var = z.var(dim=[0, 2, 3]) 
        # # Penalize if variance drops below 0.33 (ideal uniform spread in [-1, 1])
        # var_loss = F.relu(0.33 - z_var).mean()
        
        # # Combined Loss (0.1 is the weight for the divergence penalty)
        # loss = mse_loss + (0.1 * var_loss)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}")
            
    print(f"Epoch {epoch+1} Average Train Loss: {epoch_loss/len(train_loader):.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0.0
    # with torch.no_grad():
    #     for images in val_loader:
    #         images = images.to("cuda")
    #         res_feats = embedder(images)
    #         combined_feats = res_feats
    #         x_hat, z = model(combined_feats)
    #         loss = 1.0 - F.cosine_similarity(x_hat, combined_feats, dim=1).mean()
    #         val_loss += loss.item()
    with torch.no_grad():
        for images in val_loader:
            images = images.to("cuda")
            res_feats = embedder(images)
            combined_feats = res_feats
            x_hat, z = model(combined_feats)
            
            # Use the same MSE + Divergence logic for validation
            mse_loss = F.mse_loss(x_hat, combined_feats)
            z_var = z.var(dim=[0, 2, 3])
            var_loss = F.relu(0.33 - z_var).mean()
            loss = mse_loss + (0.1 * var_loss)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Average Val Loss: {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), args.save_model)
        print(f"--> Saved new best model with Val Loss: {best_val_loss:.4f} to {args.save_model}")
