"""Train a per-category nonlinear feature->3 head (PixelAutoEncoder) on top of the
INTERP (WiSE-FT, single global alpha) ResNet-101 backbone.

This learns a NONLINEAR replacement for the PCA->3 reduction: a 1x1-conv MLP that
maps the ~1792-dim ResNet feature vector at each pixel to a 3-dim latent and back,
trained to reconstruct the features (cosine loss) on nominal `train/good` images.

It deliberately reuses the SAME embedder that evaluate.py uses
(ood.embeddings.ResNetPixelEmbedder) and the SAME PixelAutoEncoder class, so the
saved .pth loads directly via evaluate.py --autoencoder_path. The only change vs
the stock trainer is that the backbone weights are overwritten with the interp
state dict (same injection as evaluate_ddad_fe.py).

Usage (env `ood`, from repo root):
  DDAD_FE_WEIGHTS=<interp.pth> CUDA_VISIBLE_DEVICES=0 \
  python all_categories_embedder/train_pixel_ae_interp.py \
      --weights <interp.pth> \
      --data_dir /data/akheirandish3/mvtec_ad/<cat>/train/good \
      --save_model all_categories_embedder/models/pixel_ae_interp/<cat>.pth \
      --latent_dim 3 --num_epochs 30
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


def set_seed(seed):
    """Make AE training reproducible: fix python/numpy/torch RNGs and force
    deterministic cuDNN. Without this, the random train/val split + weight init
    produce a different 3-d latent orientation each run -> unstable downstream AUC."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def ae_loss_fn(xh, z, feats, kind):
    """AE reconstruction loss. 'mse_cos' = MSE + (1 - cosine); 'cos' = 1 - cosine;
    'mse_var' = MSE + 0.1*ReLU(0.33 - var(z)) (train_embedder selection loss)."""
    cos = 1.0 - F.cosine_similarity(xh, feats, dim=1).mean()
    if kind == "cos":
        return cos
    if kind == "mse_cos":
        return F.mse_loss(xh, feats) + cos
    # mse_var
    z_var = z.var(dim=[0, 2, 3])
    return F.mse_loss(xh, feats) + 0.1 * F.relu(0.33 - z_var).mean()

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from ood.embeddings import ResNetPixelEmbedder       # SAME embedder eval uses
from evaluate import PixelAutoEncoder                 # SAME AE class eval loads
from all_categories_embedder.evaluate_ddad_fe import _unwrap


class ImageFolderFlat(Dataset):
    def __init__(self, root, tf):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.paths = sorted(str(p) for p in Path(root).rglob("*")
                            if p.is_file() and p.suffix.lower() in exts)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def build_embedder(weights, device):
    emb = ResNetPixelEmbedder(resnet_name="resnet101",
                              layers=("layer1", "layer2", "layer3"),
                              use_patch_context=True, patchify_size=3,
                              proj_dim_per_layer=None).to(device).eval()
    if weights:
        sd = _unwrap(torch.load(weights, map_location="cpu"))
        res = emb.extractor.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"interp backbone not fully loaded; missing {res.missing_keys[:4]}")
        print(f"  [AE-train] backbone <- {weights} (unexpected={len(res.unexpected_keys)} layer4/fc, ok)")
    else:
        print("  [AE-train] WARNING: no --weights -> ImageNet backbone")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("DDAD_FE_WEIGHTS", ""),
                    help="interp backbone state dict")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_model", required=True)
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--num_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0,
                    help="base RNG seed -> reproducible AE (deterministic split/init)")
    ap.add_argument("--n_restarts", type=int, default=1,
                    help="train this many AEs (seed, seed+1, ...) and keep the best-val one")
    ap.add_argument("--max_images", type=int, default=0,
                    help="cap dataset to this many images (0 = all); seeded subset for large sets")
    ap.add_argument("--ae_loss", default="cos", choices=["cos", "mse_cos", "mse_var"],
                    help="AE TRAINING loss. 'cos' = 1-cosine (train_embedder.py recipe).")
    ap.add_argument("--select_loss", default="mse_var", choices=["cos", "mse_cos", "mse_var"],
                    help="checkpoint-SELECTION loss (val). Default mse_var = train_embedder.py recipe.")
    args = ap.parse_args()
    device = "cuda"

    tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    ds = ImageFolderFlat(args.data_dir, tf)
    assert len(ds) > 1, f"no images under {args.data_dir}"
    if args.max_images and len(ds) > args.max_images:
        import torch as _t
        idx = _t.randperm(len(ds), generator=_t.Generator().manual_seed(args.seed))[:args.max_images].tolist()
        ds = _t.utils.data.Subset(ds, sorted(idx))
        print(f"  [AE-train] capped dataset to {len(ds)} images (seed {args.seed})")
    vN = max(1, int(0.1 * len(ds)))
    # fixed train/val split (seeded) so every run/restart sees the same data
    split_g = torch.Generator().manual_seed(args.seed)
    tr, va = random_split(ds, [len(ds) - vN, vN], generator=split_g)

    emb = build_embedder(args.weights, device)
    with torch.no_grad():
        edim = emb(torch.zeros(1, 3, 256, 256, device=device)).shape[1]
    print(f"  [AE-train] {args.data_dir}: {len(ds)} imgs  embed_dim={edim}  latent={args.latent_dim}"
          f"  seed={args.seed}  n_restarts={args.n_restarts}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
    best_overall = float("inf")
    for r in range(args.n_restarts):
        set_seed(args.seed + r)
        loader_g = torch.Generator().manual_seed(args.seed + r)
        trL = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2,
                         generator=loader_g, worker_init_fn=_seed_worker)
        vaL = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2,
                         worker_init_fn=_seed_worker)
        ae = PixelAutoEncoder(input_dim=edim, latent_dim=args.latent_dim).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=args.lr)
        best = float("inf")
        for ep in range(args.num_epochs):
            ae.train(); tot = 0.0
            for imgs in trL:
                imgs = imgs.to(device)
                with torch.no_grad():
                    feats = emb(imgs)
                xh, z = ae(feats)
                loss = ae_loss_fn(xh, z, feats, args.ae_loss)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            # select the checkpoint by the SAME loss used for training
            ae.eval(); vtot = 0.0
            with torch.no_grad():
                for imgs in vaL:
                    imgs = imgs.to(device)
                    feats = emb(imgs)
                    xh, z = ae(feats)
                    vtot += ae_loss_fn(xh, z, feats, args.select_loss).item()
            vloss = vtot / max(1, len(vaL))
            print(f"  [r{r}] ep {ep+1:02d}/{args.num_epochs}  train {tot/len(trL):.4f}  val {vloss:.4f}")
            if vloss < best:
                best = vloss
                best_sd = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
        print(f"  [r{r}] best val {best:.4f}")
        if best < best_overall:
            best_overall = best
            torch.save(best_sd, args.save_model)
            print(f"  [AE-train] new best (restart {r}) val {best_overall:.4f} -> {args.save_model}")
    print(f"  [AE-train] FINAL best val {best_overall:.4f} -> {args.save_model}")


if __name__ == "__main__":
    main()
