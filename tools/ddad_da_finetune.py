#!/usr/bin/env python
"""Fine-tune a DDAD feature extractor on clean train/good images.

This is a DVXRay-friendly version of the fine-tuning driver from the
`finetuning-code` branch. It keeps the same DDAD domain-adaptation objective:
clean images and their DDAD reconstructions should have matching ResNet
features, while the model stays anchored to the frozen ImageNet backbone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import transforms


REPO = Path(__file__).resolve().parents[1]
DDAD_DIR = REPO / "DDAD"
if str(DDAD_DIR) not in sys.path:
    sys.path.insert(0, str(DDAD_DIR))

from dataset import Dataset_maker  # noqa: E402
from feature_extractor import loss_fucntion  # noqa: E402
from reconstruction import Reconstruction  # noqa: E402
import resnet as ddad_resnet  # noqa: E402
from unet import UNetModel as DDADUNetModel  # noqa: E402


FE_BUILDERS = {
    "resnet50": ddad_resnet.resnet50,
    "resnet101": ddad_resnet.resnet101,
    "wide_resnet50_2": ddad_resnet.wide_resnet50_2,
    "wide_resnet101_2": ddad_resnet.wide_resnet101_2,
}


def strip_module_prefix(state_dict):
    if state_dict and next(iter(state_dict)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema_state_dict", "ema"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return strip_module_prefix(checkpoint)


def load_config(path, *, device, category=None, feature_extractor=None):
    cfg = OmegaConf.load(path)
    if category is not None:
        cfg.data.category = category
    if feature_extractor is not None:
        cfg.model.feature_extractor = feature_extractor
    cfg.model.device = device
    return cfg


def build_unet(cfg, checkpoint_path):
    if cfg.model.DDADS:
        unet = DDADUNetModel(
            cfg.data.image_size,
            32,
            dropout=0.3,
            n_heads=2,
            in_channels=cfg.data.input_channel,
        )
    else:
        unet = DDADUNetModel(
            cfg.data.image_size,
            64,
            dropout=0.0,
            n_heads=4,
            in_channels=cfg.data.input_channel,
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    state = unwrap_state_dict(state)
    unet.load_state_dict(state, strict=True)
    unet.to(cfg.model.device)
    unet.eval()
    for param in unet.parameters():
        param.requires_grad = False
    return unet


def build_feature_extractor(name, *, device, pretrained=True):
    if name not in FE_BUILDERS:
        raise ValueError(f"Unknown feature extractor {name!r}; choose from {sorted(FE_BUILDERS)}")
    model = FE_BUILDERS[name](pretrained=pretrained)
    model.to(device)
    return model


def plain_state_cpu(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


class LoRAConv2d(nn.Module):
    """Low-rank update for a frozen Conv2d."""

    def __init__(self, base, rank=4, alpha=1.0):
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(resnet, *, rank=4, alpha=1.0, target_layers=("layer2", "layer3")):
    lora_params = []
    for layer_name in target_layers:
        layer = getattr(resnet, layer_name)
        for block in layer:
            if hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d):
                wrapped = LoRAConv2d(block.conv2, rank=rank, alpha=alpha)
                block.conv2 = wrapped
                lora_params.extend([wrapped.lora_A.weight, wrapped.lora_B.weight])

    for param in resnet.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in resnet.parameters())
    return lora_params, n_trainable, n_total


def merge_lora_to_plain_state(lora_resnet, fe_name, device):
    plain = build_feature_extractor(fe_name, device=device, pretrained=False)
    plain_sd = plain.state_dict()
    src_sd = lora_resnet.state_dict()
    merged_weights = {}

    for name, module in lora_resnet.named_modules():
        if isinstance(module, LoRAConv2d):
            base_w = module.base.weight.data
            a = module.lora_A.weight.data
            b = module.lora_B.weight.data.squeeze(-1).squeeze(-1)
            delta = torch.einsum("or,rikl->oikl", b, a) * module.scale
            merged_weights[f"{name}.weight"] = (base_w + delta).cpu()

    out = {}
    for key in plain_sd:
        if key in merged_weights:
            out[key] = merged_weights[key]
        elif key in src_sd:
            out[key] = src_sd[key].detach().cpu()
        else:
            alt = key.replace(".conv2.", ".conv2.base.")
            out[key] = src_sd.get(alt, plain_sd[key]).detach().cpu()
    return out


def save_checkpoint(fe, *, out_dir, epoch, fe_name, lora_rank, device):
    if lora_rank > 0:
        module = fe.module if isinstance(fe, nn.DataParallel) else fe
        state = merge_lora_to_plain_state(module, fe_name, device)
    else:
        state = plain_state_cpu(fe)
    torch.save(state, out_dir / f"feat{epoch}.pth")


def count_images(path):
    return sum(1 for _ in Path(path).glob("*.png"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DDAD_DIR / "config_dvxray.yaml"))
    parser.add_argument("--category", default=None)
    parser.add_argument("--unet_ckpt", default="/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000")
    parser.add_argument("--out_dir", default="experiments/dvxray_ddad_finetune/checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature_extractor", default=None)
    parser.add_argument("--da_epochs", type=int, default=None)
    parser.add_argument("--da_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w_da", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=None, help="debug cap per epoch")
    parser.add_argument("--limit_train_images", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--parallel", action="store_true", help="wrap feature extractor in DataParallel")
    parser.add_argument("--dry_run", action="store_true", help="load models/data, then exit")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(
        args.config,
        device=args.device,
        category=args.category,
        feature_extractor=args.feature_extractor,
    )
    fe_name = str(cfg.model.feature_extractor)
    if args.da_epochs is not None:
        cfg.model.DA_epochs = args.da_epochs
    if args.da_batch_size is not None:
        cfg.data.DA_batch_size = args.da_batch_size
    if args.num_workers is not None:
        cfg.model.num_workers = args.num_workers
    if args.w_da is not None:
        cfg.model.w_DA = args.w_da

    if cfg.data.DA_batch_size % 2 != 0:
        raise ValueError("DA_batch_size must be even because batches are split into input/target halves")

    train_path = Path(cfg.data.data_dir) / cfg.data.category / "train" / "good"
    train_image_count = count_images(train_path)
    if train_image_count == 0:
        raise RuntimeError(f"No training images found at {train_path}")

    print(f"config: {args.config}")
    print(f"category: {cfg.data.category}")
    print(f"train/good: {train_path} ({train_image_count} pngs)")
    print(f"unet checkpoint: {args.unet_ckpt}")
    print(f"feature extractor: {fe_name}")
    print(f"out_dir: {out_dir}")
    print(f"device: {args.device}")

    unet = build_unet(cfg, args.unet_ckpt)
    fe = build_feature_extractor(fe_name, device=args.device, pretrained=True)
    frozen = build_feature_extractor(fe_name, device=args.device, pretrained=True)
    frozen.eval()
    for param in frozen.parameters():
        param.requires_grad = False

    trainable_params = list(fe.parameters())
    n_total = sum(p.numel() for p in fe.parameters())
    n_trainable = n_total
    if args.lora_rank > 0:
        trainable_params, n_trainable, n_total = inject_lora(
            fe, rank=args.lora_rank, alpha=args.lora_alpha
        )
        print(
            f"LoRA rank={args.lora_rank}: trainable {n_trainable:,}/{n_total:,} "
            f"({100.0 * n_trainable / n_total:.4f}%)"
        )

    if args.parallel:
        fe = nn.DataParallel(fe)
        frozen = nn.DataParallel(frozen)

    save_checkpoint(
        frozen,
        out_dir=out_dir,
        epoch=0,
        fe_name=fe_name,
        lora_rank=0,
        device=args.device,
    )

    train_dataset = Dataset_maker(
        root=cfg.data.data_dir,
        category=cfg.data.category,
        config=cfg,
        is_train=True,
    )
    if args.limit_train_images is not None:
        n = min(args.limit_train_images, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(n))

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.DA_batch_size,
        shuffle=True,
        num_workers=cfg.model.num_workers,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("Training loader is empty; reduce DA_batch_size or add training images")

    log = {
        "config": os.path.abspath(args.config),
        "category": str(cfg.data.category),
        "data_dir": str(cfg.data.data_dir),
        "train_path": str(train_path),
        "n_train_images": int(len(train_dataset)),
        "unet_ckpt": os.path.abspath(args.unet_ckpt),
        "feature_extractor": fe_name,
        "da_epochs": int(cfg.model.DA_epochs),
        "da_batch_size": int(cfg.data.DA_batch_size),
        "max_batches": args.max_batches,
        "limit_train_images": args.limit_train_images,
        "lr": args.lr,
        "w_DA": float(cfg.model.w_DA),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "n_total": int(n_total),
        "n_trainable": int(n_trainable),
        "pct_trainable": round(100.0 * n_trainable / n_total, 6),
        "epochs": [],
    }
    with open(out_dir / "da_log.json", "w") as handle:
        json.dump(log, handle, indent=2)

    if args.dry_run:
        print("dry run complete; models, dataset, and feat0.pth loaded/saved")
        return

    recon = Reconstruction(unet, cfg)
    norm = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    fe.train()
    if args.lora_rank > 0:
        freeze_batchnorm(fe)

    for epoch in range(int(cfg.model.DA_epochs)):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if args.lora_rank > 0:
            freeze_batchnorm(fe)

        t0 = time.time()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(loader, start=1):
            images = batch[0].to(args.device)
            half = images.shape[0] // 2
            target = images[:half]
            inputs = images[half : half * 2]

            x0 = recon(inputs, target, cfg.model.w_DA)[-1].to(args.device)
            x0 = norm(x0)
            target_norm = norm(target)

            recon_features = fe(x0)
            target_features = fe(target_norm)
            target_frozen = frozen(target_norm)
            recon_frozen = frozen(x0)
            loss = loss_fucntion(recon_features, target_features, target_frozen, recon_frozen, cfg)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
                print(
                    f"epoch {epoch + 1}/{cfg.model.DA_epochs} "
                    f"batch {step}/{len(loader)} loss={loss.item():.6f}"
                )
            if args.max_batches is not None and step >= args.max_batches:
                break

        epoch_loss = running / max(n_batches, 1)
        seconds = time.time() - t0
        save_checkpoint(
            fe,
            out_dir=out_dir,
            epoch=epoch + 1,
            fe_name=fe_name,
            lora_rank=args.lora_rank,
            device=args.device,
        )
        entry = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "batches": n_batches,
            "seconds": round(seconds, 2),
        }
        if torch.cuda.is_available():
            entry["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        log["epochs"].append(entry)
        with open(out_dir / "da_log.json", "w") as handle:
            json.dump(log, handle, indent=2)
        print(
            f"epoch {epoch + 1}/{cfg.model.DA_epochs} done: "
            f"loss={epoch_loss:.6f}, batches={n_batches}, seconds={seconds:.1f}"
        )

    print(f"fine-tuning complete; checkpoints are in {out_dir}")


if __name__ == "__main__":
    main()
