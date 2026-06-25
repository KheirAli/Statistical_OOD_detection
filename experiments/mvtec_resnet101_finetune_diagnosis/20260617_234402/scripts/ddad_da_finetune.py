"""
DDAD-style ResNet-101 feature-extractor domain-adaptation fine-tuning,
per category, saving a checkpoint EVERY epoch.

Reuses the DDAD loss (feature_extractor.loss_fucntion) and the DDAD diffusion
Reconstruction unmodified. Supports:
  - full fine-tuning (--lora_rank 0)
  - LoRA parameter-constrained fine-tuning (--lora_rank r)
  - training-time control via --da_epochs

Outputs (under --out_dir):
  feat0.pth                     # pretrained ImageNet ResNet-101 (alpha=0 anchor)
  feat1.pth ... featN.pth       # after each epoch (plain resnet101 state dict)
  da_log.json                   # per-epoch loss, timing, param counts
For LoRA, featK are the *merged* plain resnet101 weights so eval is identical.
"""
import os, sys, time, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import ddad_diag as D
from reconstruction import Reconstruction
from feature_extractor import loss_fucntion
from dataset import Dataset_maker
from torchvision import transforms
import torch.nn as nn


def freeze_batchnorm(model):
    """Put every BatchNorm in eval() so its running stats do NOT drift.
    Required for LoRA/parameter-constrained FT: with the conv weights frozen,
    letting BN running_mean/var adapt to the new domain corrupts the features
    (frozen convs expect the pretrained BN statistics)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--unet_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--da_epochs", type=int, default=8)
    ap.add_argument("--da_batch_size", type=int, default=16)
    ap.add_argument("--w_da", type=float, default=None, help="override config w_DA")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lora_rank", type=int, default=0, help="0 = full fine-tune")
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--train_scope", default="full", choices=["full", "last_layer"],
                    help="full = all weights; last_layer = freeze backbone, train only layer3 "
                         "(the deepest stage the detector uses). Ignored when --lora_rank>0.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = D.load_config(args.category, device=args.device)
    if args.w_da is not None:
        cfg.model.w_DA = args.w_da
    cfg.data.DA_batch_size = args.da_batch_size

    unet = D.build_unet(cfg, args.unet_ckpt)
    unet.eval()

    # trainable + frozen FE (DDAD anchor). resnet101 returns feature lists.
    fe = D.build_resnet101(args.device, pretrained=True, parallel=True)
    frozen = D.build_resnet101(args.device, pretrained=True, parallel=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad = False

    log = {"category": args.category, "lora_rank": args.lora_rank,
           "da_epochs": args.da_epochs, "lr": args.lr, "w_DA": float(cfg.model.w_DA),
           "epochs": []}

    # LoRA injection (on the underlying module of the DataParallel wrapper)
    if args.lora_rank > 0:
        base = fe.module
        base, lora_params, n_tr, n_tot = D.inject_lora(base, r=args.lora_rank,
                                                       alpha=args.lora_alpha)
        fe = torch.nn.DataParallel(base.to(args.device))
        trainable = [p for p in fe.parameters() if p.requires_grad]
        log["n_trainable"] = int(n_tr); log["n_total"] = int(n_tot)
        log["pct_trainable"] = round(100.0 * n_tr / n_tot, 4)
        print(f"[LoRA r={args.lora_rank}] trainable {n_tr:,}/{n_tot:,} "
              f"({100.0*n_tr/n_tot:.3f}%)")
    elif args.train_scope == "last_layer":
        # freeze the whole backbone, train only the LAST block of layer3 (the
        # final layer producing the deepest features the detector uses) -> a true
        # lightweight "last layer" adaptation, not the whole stage.
        base = fe.module if hasattr(fe, "module") else fe
        for p in base.parameters():
            p.requires_grad = False
        for p in base.layer3[-1].parameters():
            p.requires_grad = True
        trainable = [p for p in fe.parameters() if p.requires_grad]
        n_tr = sum(p.numel() for p in trainable)
        n_tot = sum(p.numel() for p in fe.parameters())
        log["n_trainable"] = int(n_tr); log["n_total"] = int(n_tot)
        log["pct_trainable"] = round(100.0 * n_tr / n_tot, 4)
        log["train_scope"] = "last_layer(layer3[-1])"
        print(f"[last_layer] trainable {n_tr:,}/{n_tot:,} ({100.0*n_tr/n_tot:.3f}%) = layer3 last block")
    else:
        trainable = list(fe.parameters())
        n_tot = sum(p.numel() for p in fe.parameters())
        log["n_trainable"] = int(n_tot); log["n_total"] = int(n_tot)
        log["pct_trainable"] = 100.0

    # save feat0 = pretrained anchor (plain resnet101 state dict)
    if args.lora_rank > 0:
        # for lora, feat0 is the clean pretrained backbone (no lora delta -> B is 0 anyway)
        feat0 = D.fe_state_cpu(frozen)
    else:
        feat0 = D.fe_state_cpu(frozen)
    torch.save(feat0, os.path.join(args.out_dir, "feat0.pth"))

    # data
    train_ds = Dataset_maker(root=cfg.data.data_dir, category=cfg.data.category,
                             config=cfg, is_train=True)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.data.DA_batch_size,
                                         shuffle=True, num_workers=cfg.model.num_workers,
                                         drop_last=True)
    log["n_train_images"] = len(train_ds)
    print(f"[{args.category}] train images: {len(train_ds)}  batches/epoch: {len(loader)}")

    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    recon = Reconstruction(unet, cfg)

    fe.train()
    if args.lora_rank > 0:
        freeze_batchnorm(fe)   # keep pretrained BN stats; only LoRA A/B adapt
    for epoch in range(args.da_epochs):
        if args.lora_rank > 0:
            freeze_batchnorm(fe)
        t0 = time.time(); ep_loss = 0.0; nb = 0
        for batch in loader:
            half = batch[0].shape[0] // 2
            target = batch[0][:half].to(args.device)
            inp = batch[0][half:].to(args.device)
            x0 = recon(inp, target, cfg.model.w_DA)[-1].to(args.device)
            x0 = transform(x0); target = transform(target)
            r_fe = fe(x0); t_fe = fe(target)
            t_frz = frozen(target); r_frz = frozen(x0)
            loss = loss_fucntion(r_fe, t_fe, t_frz, r_frz, cfg)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        ep_loss /= max(nb, 1)
        dt = time.time() - t0
        print(f"[{args.category}] epoch {epoch+1}/{args.da_epochs} "
              f"loss={ep_loss:.5f} ({dt:.1f}s)")

        # save checkpoint as a plain resnet101 state dict
        if args.lora_rank > 0:
            plain_sd = D.merge_lora_to_plain_resnet(fe.module, args.device)
        else:
            plain_sd = D.fe_state_cpu(fe)
        torch.save(plain_sd, os.path.join(args.out_dir, f"feat{epoch+1}.pth"))
        log["epochs"].append({"epoch": epoch + 1, "loss": ep_loss,
                              "seconds": round(dt, 1)})
        with open(os.path.join(args.out_dir, "da_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print(f"[{args.category}] DA done. checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
