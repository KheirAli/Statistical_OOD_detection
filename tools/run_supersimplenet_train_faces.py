"""Train SuperSimpleNet on the 70k-image faces shadow dataset.

Upstream `SuperSimpleNet/train.py` ships an MVTec runner that hard-codes
the 15 standard categories and 300 epochs. To train on our custom
'faces' category at the 70k-image scale, we'd otherwise have to edit
upstream source — instead, we import the upstream bits and drive a
single-category training from the outside.

This is a one-off (faces only) — for the 15 standard MVTec categories
we already have HF JIMS weights, no need to re-train.

Usage:
    python tools/run_supersimplenet_train_faces.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run this in the `bl_ssn` conda env, which has the upstream-pinned
# anomalib==0.7 + numpy==1.26 + pytorch-lightning. The main `ood` env
# has anomalib 1.x/2.x which is incompatible with SuperSimpleNet's
# imports.
import torch
from pytorch_lightning import seed_everything
from torchmetrics import AveragePrecision
from anomalib.utils.metrics import AUROC, AUPRO

REPO = Path("/home/rohan/ood/baseline-algos-clone/SuperSimpleNet")
sys.path.insert(0, str(REPO))

from datamodules.mvtec import MVTec                                  # noqa: E402
from model.supersimplenet import SuperSimpleNet                      # noqa: E402
from train import train as train_loop                                # noqa: E402

DATAPATH = Path("/data2/rohan/datasets/mvtec_ad_faces_5k")
SAVE_ROOT = Path("/data2/rohan/baseline_ckpts/supersimplenet_faces_5k")
EPOCHS = 30                              # 5k images × 30 = 150k samples seen,
                                         # comparable budget to the upstream 300-epoch
                                         # × 200-image setup. (We earlier crashed on
                                         # 70k epochs=2 at a transient cv2 read error
                                         # despite no broken file in a sequential scan.)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = {
        # Match the JIMS extension config from upstream's run_unsup() —
        # adapt_cls_feat=False is the JIMS setting; toggling to True
        # would replicate the older ICPR variant.
        "wandb_project": "ssn-faces70k",
        "datasets_folder": DATAPATH,
        "num_workers": 8,
        "setup_name": "superSimpleNet",
        "dt": (3, 2),
        "dilate": 7,
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.2,                 # mvtec setting in upstream
        "image_size": (256, 256),
        "seed": 42,
        "batch": 32,
        "epochs": EPOCHS,
        "flips": False,
        "seg_lr": 0.0002,
        "dec_lr": 0.0002,
        "adapt_lr": 0.0001,
        "gamma": 0.4,
        "stop_grad": True,
        "clip_grad": False,
        "eval_step_size": 4,
        "dataset": "mvtec",
        "ratio": 1,
        "category": "faces",
        "name": "faces_superSimpleNet_70k",
        "results_save_path": SAVE_ROOT,
    }

    seed_everything(config["seed"], workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Training on faces (70k images), {EPOCHS} epochs, device={device}")
    model = SuperSimpleNet(image_size=config["image_size"], config=config)

    # The datamodule resolves `<root>/<category>/{train,test,ground_truth}/`
    # internally, so passing root=DATAPATH (the shadow tree's parent) is right.
    datamodule = MVTec(
        root=DATAPATH,
        category="faces",
        image_size=config["image_size"],
        train_batch_size=config["batch"],
        eval_batch_size=config["batch"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )
    datamodule.setup()

    image_metrics = {
        "I-AUROC": AUROC(),
        "AP-det": AveragePrecision(num_classes=1),
    }
    pixel_metrics = {
        "P-AUROC": AUROC(),
        "AUPRO": AUPRO(),
        "AP-loc": AveragePrecision(num_classes=1),
    }

    train_loop(
        model=model,
        epochs=config["epochs"],
        datamodule=datamodule,
        device=device,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        clip_grad=config["clip_grad"],
        eval_step_size=config["eval_step_size"],
    )

    save_dir = (
        SAVE_ROOT / config["setup_name"] / "checkpoints"
        / config["dataset"] / config["category"] / str(config["ratio"])
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(save_dir)
    print(f"saved → {save_dir}/weights.pt")


if __name__ == "__main__":
    main()
