"""
Shared library for the MVTec ResNet-101 fine-tuning diagnosis.

Reuses the vendored DDAD pipeline (DDAD/) unmodified:
  - reconstruction.Reconstruction   (diffusion reconstructor)
  - anomaly_map.heat_map            (FE-based anomaly map)
  - metrics.Metric                  (image/pixel AUROC, PRO, AP)
  - feature_extractor.loss_fucntion (DDAD domain-adaptation loss)
  - resnet.resnet101                (modified ResNet returning feature lists)
  - dataset.Dataset_maker           (MVTec loader)

Key idea exploited everywhere: in DDAD detection the diffusion reconstruction
x0 and the input do NOT depend on the feature extractor (only heat_map's
feature_distance term uses the FE). So we reconstruct the test set ONCE, cache
(x0, input, gt, label), and evaluate any number of FE variants cheaply.
"""
import os, sys, time, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- make the vendored DDAD package importable (its modules import each other
#     with bare names, e.g. `from dataset import *`) ---
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DDAD_DIR = os.path.join(REPO, "DDAD")
assert os.path.isdir(DDAD_DIR), f"DDAD dir not found at {DDAD_DIR}"
if DDAD_DIR not in sys.path:
    sys.path.insert(0, DDAD_DIR)

from omegaconf import OmegaConf
from reconstruction import Reconstruction          # noqa: E402
from anomaly_map import heat_map                    # noqa: E402
from metrics import Metric                          # noqa: E402
from feature_extractor import loss_fucntion         # noqa: E402
from dataset import Dataset_maker                   # noqa: E402
import resnet as ddad_resnet                        # noqa: E402
from unet import UNetModel as DDADUNetModel         # noqa: E402

CATEGORIES = ["carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
              "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
              "transistor", "zipper"]


# --------------------------------------------------------------------------- #
# config / model construction
# --------------------------------------------------------------------------- #
def load_config(category, device="cuda", config_path=None):
    if config_path is None:
        config_path = os.path.join(DDAD_DIR, "config.yaml")
    cfg = OmegaConf.load(config_path)
    cfg.data.category = category
    cfg.model.device = device
    cfg.model.feature_extractor = "resnet101"
    return cfg


def build_unet(cfg, unet_ckpt):
    """Build the DDAD UNet and load a checkpoint (DataParallel state dict)."""
    if cfg.model.DDADS:
        unet = DDADUNetModel(cfg.data.image_size, 32, dropout=0.3, n_heads=2,
                             in_channels=cfg.data.input_channel)
    else:
        unet = DDADUNetModel(cfg.data.image_size, 64, dropout=0.0, n_heads=4,
                             in_channels=cfg.data.input_channel)
    state = torch.load(unet_ckpt, map_location="cpu")
    if isinstance(state, dict) and len(state) and next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    unet.load_state_dict(state, strict=True)
    unet = unet.to(cfg.model.device)
    unet.eval()
    return unet


def build_resnet101(device, pretrained=True, parallel=True):
    """Plain DDAD resnet101 (returns [layer1, layer2, layer3] feature lists)."""
    fe = ddad_resnet.resnet101(pretrained=pretrained)
    fe = fe.to(device)
    if parallel:
        fe = torch.nn.DataParallel(fe)
    return fe


def _strip_module(sd):
    if len(sd) and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_fe_from_state(device, state_dict_or_path, parallel=True):
    """Build a resnet101 and load a (possibly DataParallel) state dict / path."""
    if isinstance(state_dict_or_path, str):
        sd = torch.load(state_dict_or_path, map_location="cpu")
    else:
        sd = state_dict_or_path
    sd = _strip_module(sd)
    fe = ddad_resnet.resnet101(pretrained=False)
    fe.load_state_dict(sd, strict=True)
    fe = fe.to(device)
    if parallel:
        fe = torch.nn.DataParallel(fe)
    fe.eval()
    return fe


def fe_state_cpu(fe):
    """Return a plain (no module.) cpu float state dict for a (DataParallel) FE."""
    sd = fe.module.state_dict() if isinstance(fe, nn.DataParallel) else fe.state_dict()
    return {k: v.detach().cpu() for k, v in sd.items()}


# --------------------------------------------------------------------------- #
# reconstruction cache
# --------------------------------------------------------------------------- #
def reconstruct_and_cache(cfg, unet, cache_path, seed=42, limit=None):
    """Reconstruct the whole test set once and cache tensors to disk.

    Returns dict with lists of cpu tensors: inputs, recons, gts, labels.
    """
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    torch.manual_seed(seed); np.random.seed(seed)
    test_ds = Dataset_maker(root=cfg.data.data_dir, category=cfg.data.category,
                            config=cfg, is_train=False)
    # optional: drop curated/duplicate test subdirs (e.g. cable/test/combined)
    excl = list(cfg.data.get("exclude_test_subdirs", []) or [])
    if excl:
        before = len(test_ds.image_files)
        test_ds.image_files = [f for f in test_ds.image_files
                               if os.path.basename(os.path.dirname(f)) not in excl]
        print(f"[{cfg.data.category}] excluded test subdirs {excl}: "
              f"{before} -> {len(test_ds.image_files)} images")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.data.test_batch_size,
                                         shuffle=False, num_workers=cfg.model.num_workers,
                                         drop_last=False)
    recon = Reconstruction(unet, cfg)
    inputs, recons, gts, labels = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (inp, gt, lab) in enumerate(loader):
            if limit is not None and i >= limit:
                break
            inp = inp.to(cfg.model.device)
            x0 = recon(inp, inp, cfg.model.w)[-1]
            inputs.append(inp.cpu())
            recons.append(x0.cpu())
            gts.append(gt.cpu())
            labels.extend([0 if l == "good" else 1 for l in lab])
    cache = {"inputs": inputs, "recons": recons, "gts": gts, "labels": labels,
             "recon_seconds": time.time() - t0, "n": len(inputs)}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache, cache_path)
    return cache


# --------------------------------------------------------------------------- #
# evaluation of a single FE on cached reconstructions
# --------------------------------------------------------------------------- #
def _seg_metrics_at_best_f1(maps_norm, gts_bin):
    """maps_norm: (N,H,W) in [0,1]; gts_bin: (N,H,W) {0,1}. Pick threshold
    maximizing pixel F1; return dict of dice/iou/f1/precision/recall + thr."""
    flat_m = maps_norm.reshape(-1)
    flat_g = gts_bin.reshape(-1).astype(np.uint8)
    if flat_g.sum() == 0:
        return dict(dice=float("nan"), iou=float("nan"), f1=float("nan"),
                    precision=float("nan"), recall=float("nan"), seg_threshold=float("nan"))
    # candidate thresholds (quantiles of anomaly scores for efficiency)
    qs = np.linspace(0.5, 0.999, 60)
    thrs = np.quantile(flat_m, qs)
    best = dict(f1=-1)
    P = flat_g.sum()
    order = None
    for t in thrs:
        pred = flat_m >= t
        tp = np.logical_and(pred, flat_g == 1).sum()
        fp = np.logical_and(pred, flat_g == 0).sum()
        fn = P - tp
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        iou = tp / (tp + fp + fn + 1e-12)
        if f1 > best["f1"]:
            best = dict(dice=float(f1), iou=float(iou), f1=float(f1),
                        precision=float(prec), recall=float(rec), seg_threshold=float(t))
    return best


def evaluate_fe(cfg, cache, fe, compute_pro=True, tag=""):
    """Evaluate one feature extractor on cached reconstructions.

    Returns a dict of metrics. Reuses DDAD heat_map + DDAD Metric for the
    native metrics; adds dice/iou/f1/precision/recall + auprc.
    """
    device = cfg.model.device
    fe.eval()
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    t0 = time.time()

    anomaly_map_list, gt_list, labels_list, predictions = [], [], [], []
    resize = torch.nn.Upsample(size=(256, 256), mode="bilinear", align_corners=True)
    with torch.no_grad():
        for inp, x0, gt, lab in zip(cache["inputs"], cache["recons"], cache["gts"],
                                    cache["labels"]):
            inp = inp.to(device); x0 = x0.to(device)
            amap = heat_map(x0, inp, fe, cfg)            # (1,1,256,256)
            amap = amap.detach().cpu()
            if gt.shape[-2:] != (256, 256):
                gt = F.interpolate(gt.float(), size=(256, 256), mode="nearest")
            anomaly_map_list.append(amap)
            gt_list.append(gt.cpu())
            labels_list.append(int(lab))
            predictions.append(float(amap.max()))

    # ---- native DDAD metrics ----
    metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, cfg)
    out = {"tag": tag}
    try:
        out["image_auroc"] = float(metric.image_auroc())
    except Exception as e:
        out["image_auroc"] = float("nan"); print(f"[warn] image_auroc: {e}")
    try:
        out["pixel_auroc"] = float(metric.pixel_auroc())
    except Exception as e:
        out["pixel_auroc"] = float("nan"); print(f"[warn] pixel_auroc: {e}")
    try:
        out["auprc"] = float(metric.pixel_ap_per_image_mean())
    except Exception as e:
        out["auprc"] = float("nan"); print(f"[warn] auprc: {e}")
    if compute_pro:
        try:
            out["pro"] = float(metric.pixel_pro())
        except Exception as e:
            out["pro"] = float("nan"); print(f"[warn] pro: {e}")
    else:
        out["pro"] = float("nan")

    # ---- segmentation metrics (added) ----
    maps = torch.cat(anomaly_map_list, dim=0).squeeze(1).numpy().astype(np.float32)  # (N,H,W)
    gts = torch.cat(gt_list, dim=0).squeeze(1).numpy()
    gts_bin = (gts > 0.5).astype(np.uint8)
    mn, mx = maps.min(), maps.max()
    maps_norm = (maps - mn) / (mx - mn + 1e-8)
    seg = _seg_metrics_at_best_f1(maps_norm, gts_bin)
    out.update(seg)

    out["runtime_sec"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        out["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    else:
        out["gpu_memory_mb"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# WiSE-FT linear weight interpolation
# --------------------------------------------------------------------------- #
def interpolate_state_dicts(sd_a, sd_b, alpha):
    """theta = (1-alpha)*sd_a + alpha*sd_b, matching name+shape, float only.
    Non-float tensors are copied from sd_b (the fine-tuned side)."""
    sd_a = _strip_module(sd_a); sd_b = _strip_module(sd_b)
    out = {}
    for k in sd_b:
        vb = sd_b[k]
        if k in sd_a and sd_a[k].shape == vb.shape and torch.is_floating_point(vb):
            out[k] = (1 - alpha) * sd_a[k].float() + alpha * vb.float()
            out[k] = out[k].to(vb.dtype)
        else:
            out[k] = vb.clone()
    return out


# --------------------------------------------------------------------------- #
# Minimal LoRA for ResNet conv layers
# --------------------------------------------------------------------------- #
class LoRAConv2d(nn.Module):
    """Wrap a frozen nn.Conv2d with a trainable low-rank update.
    y = conv(x) + scale * B(A(x)), A,B implemented as 1x1->kxk decomposition
    via two convs. Frozen base + small trainable A,B."""
    def __init__(self, base: nn.Conv2d, r=4, alpha=1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        # A: in_ch -> r with same kernel/stride/padding; B: r -> out_ch 1x1
        self.lora_A = nn.Conv2d(base.in_channels, r, kernel_size=base.kernel_size,
                                stride=base.stride, padding=base.padding,
                                dilation=base.dilation, groups=1, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(resnet, r=4, alpha=1.0, targets=("layer2", "layer3")):
    """Replace 3x3 conv2 in Bottleneck blocks of the target layers with LoRA.
    Returns (model, lora_params, n_trainable, n_total)."""
    lora_params = []
    for lname in targets:
        layer = getattr(resnet, lname)
        for block in layer:
            if hasattr(block, "conv2") and isinstance(block.conv2, nn.Conv2d):
                wrapped = LoRAConv2d(block.conv2, r=r, alpha=alpha)
                block.conv2 = wrapped
                lora_params += [wrapped.lora_A.weight, wrapped.lora_B.weight]
    # freeze everything except lora params
    for p in resnet.parameters():
        p.requires_grad = False
    for p in lora_params:
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in resnet.parameters())
    return resnet, lora_params, n_trainable, n_total


def merge_lora_to_plain_resnet(lora_resnet, device):
    """Fold LoRA deltas into base conv weights and return a PLAIN resnet101
    state dict (no LoRA modules) so the eval harness can load it normally.

    For a conv with kernel K, weight W (out,in,kh,kw): the LoRA branch is
    B(A(x)) with A weight (r,in,kh,kw) and B weight (out,r,1,1). The composed
    linear map equals a conv with weight Wd where
        Wd[o,i,:,:] = scale * sum_j B[o,j,0,0] * A[j,i,:,:]
    (B is 1x1, so it mixes channels only). We add Wd to W."""
    plain = ddad_resnet.resnet101(pretrained=False).to(device)
    # start from the lora model's base weights
    src = lora_resnet
    plain_sd = plain.state_dict()
    # build a mapping from lora model state by walking modules
    new_sd = {}
    for name, module in src.named_modules():
        if isinstance(module, LoRAConv2d):
            W = module.base.weight.data.clone()                      # (out,in,kh,kw)
            A = module.lora_A.weight.data                            # (r,in,kh,kw)
            B = module.lora_B.weight.data.squeeze(-1).squeeze(-1)    # (out,r)
            Wd = torch.einsum("or,rikl->oikl", B, A) * module.scale
            new_sd[name + ".weight"] = (W + Wd).cpu()
    # full plain state dict: take from src where possible
    src_sd = src.state_dict()
    out_sd = {}
    for k in plain_sd:
        if k in new_sd:
            out_sd[k] = new_sd[k]
        else:
            # map lora base param names: lora wrappers renamed conv2 -> conv2.base
            cand = k
            if cand in src_sd:
                out_sd[k] = src_sd[cand].cpu()
            else:
                # try inserting .base for conv2 weights that became LoRAConv2d.base
                alt = k.replace(".conv2.", ".conv2.base.")
                if alt in src_sd:
                    out_sd[k] = src_sd[alt].cpu()
                else:
                    out_sd[k] = plain_sd[k].cpu()
    return out_sd
