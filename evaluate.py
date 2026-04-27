# #!/usr/bin/env python3
# """Evaluate OOD detection pipeline.

# Sampling is handled by tools/run_ddad_reconstruction.py or
# tools/run_ddad_dps_sampling.py. evaluate.py only scores existing recons.

# Usage:
#     python evaluate.py --config configs/experiment_ddad_native.yaml \\
#         --skip_sampling --no_plots --scorer typical_set \\
#         --n_pca 5 --bins_pca 16 --sample_name samples_000
#     python evaluate.py --config configs/experiment_ddad_native.yaml \\
#         --skip_sampling --no_plots --scorer local_gaussian --sigma_rohan 0.1 \\
#         --sample_names samples_000 samples_001 samples_002 ... samples_010
# """

# import argparse
# import json
# import os
# import subprocess
# import sys
# from copy import deepcopy

# import numpy as np
# import yaml

# from ood.data import load_gt_mask, load_label_image, load_reconstructions, load_superpixel_mask, parse_sample_id
# from ood.superpixels import recursive_subdivide
# from ood.embeddings import ResNetPixelEmbedder, embed_and_project
# from ood.scoring import compute_delta_map_gpu
# from ood.scoring_local_gaussian import compute_delta_map_local_gaussian
# from ood.metrics import evaluate_delta_map
# # Note: ood.sampler and ood.baselines were removed in the push-ready cleanup.
# # Reconstructions are now generated via tools/run_ddad_reconstruction.py or
# # tools/run_ddad_dps_sampling.py. Always pass --skip_sampling to evaluate.py.


# def load_config(path: str) -> dict:
#     with open(path) as f:
#         return yaml.safe_load(f)


# def _run_single_sample(cfg: dict) -> dict:
#     """Run evaluation on a single sample. Returns metrics dict."""
#     import torch

#     data_cfg = cfg["data"]
#     sample_name = data_cfg["sample_name"]
#     sample_id = parse_sample_id(sample_name)
#     device = cfg["embeddings"].get("device", "cuda")

#     # Per-sample figures directory so each image gets its own superpixel mask
#     figures_dir = os.path.join(data_cfg["figures_dir"], sample_name)

#     print(f"\n=== OOD Evaluation: {sample_name} ===")
#     print("Sampling disabled, using existing results.")

#     # Ensure per-sample superpixel mask exists (generate if missing)
#     mask_path = os.path.join(figures_dir, "mask.png")
#     if not os.path.exists(mask_path):
#         image_dir = data_cfg["image_dir"]
#         input_image = os.path.join(image_dir, f"{sample_id}.png")
#         os.makedirs(figures_dir, exist_ok=True)
#         if os.path.exists(input_image):
#             print(f"  Generating superpixels for {sample_name}...")
#             subprocess.run(
#                 [sys.executable, "super_pixel_generation.py",
#                  f"--input_image={input_image}",
#                  f"--output_dir={figures_dir}"],
#                 check=True,
#             )

#     # Stage 1: Load data
#     print("Loading data...")
#     recon_all = load_reconstructions(
#         results_dir=data_cfg["results_dir"],
#         sample_name=sample_name,
#         test_origin=data_cfg["test_origin"],
#         num_patches=cfg["sampling"].get("num_patches", 24),
#         bottom_suffix=data_cfg["bottom_suffix"],
#     )
#     label_image = load_label_image(
#         results_dir=data_cfg["results_dir"],
#         sample_name=sample_name,
#         test_origin=data_cfg["test_origin"],
#         bottom_suffix=data_cfg["bottom_suffix"],
#     )
#     sp_mask = load_superpixel_mask(figures_dir)
#     gt_mask = load_gt_mask(
#         path_template=data_cfg["gt_mask"]["path"],
#         sample=sample_id,
#         downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
#     )

#     # Stage 2: Superpixel refinement
#     sp_cfg = cfg["superpixels"]
#     labels_fine, final_ids, parent_map = recursive_subdivide(
#         img=label_image, labels=sp_mask,
#         var_threshold=sp_cfg["var_threshold"], min_pixels=sp_cfg["min_pixels"],
#         max_sub=sp_cfg["max_sub"], max_depth=sp_cfg["max_depth"],
#         compactness=sp_cfg["compactness"], alpha_grad=sp_cfg["alpha_grad"],
#         target_size=sp_cfg["target_size"],
#     )
#     print(f"  {len(np.unique(sp_mask))} -> {len(final_ids)} superpixels")

#     # Stage 3+4: Embed + PCA
#     embed_cfg = cfg["embeddings"]
#     embedder = ResNetPixelEmbedder(
#         resnet_name=embed_cfg["backbone"],
#         layers=tuple(embed_cfg["layers"]),
#         use_patch_context=embed_cfg["use_patch_context"],
#         patchify_size=embed_cfg.get("patchify_size", 3),
#         proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer"),
#     ).to(device).eval()

#     embed_result = embed_and_project(
#         embedder=embedder, label_image=label_image,
#         images_recon_all=recon_all, n_pca=cfg["pca"]["n_components"],
#         device=device,
#     )
#     del embedder
#     torch.cuda.empty_cache()

#     # Stage 5: Scoring
#     score_cfg = cfg["scoring"]
#     algorithm = score_cfg.get("algorithm", "typical_set")
#     if algorithm == "local_gaussian":
#         sigma_rohan = score_cfg.get("sigma_rohan")
#         if sigma_rohan is None:
#             sigma_path = os.path.join(data_cfg["results_dir"], "sigma.txt")
#             if os.path.exists(sigma_path):
#                 with open(sigma_path) as f:
#                     sigma_rohan = float(f.read().strip())
#                 print(f"  sigma_rohan={sigma_rohan} (from {sigma_path})")
#             else:
#                 raise ValueError("local_gaussian scorer needs --sigma_rohan or a sigma.txt under results_dir")
#         # Use base SP mask (not refined) for local_gaussian: the scorer needs
#         # D = n_sp * C to be tractable relative to n_realizations.  With 5636
#         # refined SPs (D=16908) and rank ≤ n_realizations-1, the precision matrix
#         # has near-zero coverage → metric collapses to 0.5.  Base mask (~44 SPs,
#         # D=132) gives full-rank covariance with n_realizations=1000.
#         lg_labels = sp_mask   # base mask, not labels_fine
#         lg_parent = {int(sid): [int(sid)] for sid in np.unique(sp_mask)}
#         delta_map, _info, labels_used = compute_delta_map_local_gaussian(
#             labels_fine=lg_labels, parent_map=lg_parent,
#             images_recon_all=recon_all, label_image=label_image,
#             sigma=sigma_rohan,
#             n_realizations=score_cfg.get("n_realizations", 1000),
#             min_pixels=score_cfg["min_pixels"],
#             metric=score_cfg.get("lg_metric", "typicality_unsigned"),
#         )
#     else:
# #         category = getattr(args, "category", None) or data_cfg.get("category", "cable")
# # # Then:
# #         gray_scale=(category == "CT")
#         gray_scale = data_cfg.get("gray_scale", False)
#         # delta_map, _info, labels_used = compute_delta_map(
#         #     labels_fine=labels_fine, parent_map=parent_map,
#         #     images_recon_all=recon_all, pca_feats_recon=embed_result["pca_feats_recon"],
#         #     label_image=label_image, label_pca_map=embed_result["label_pca_map"],
#         #     bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
#         #     smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
#         #     use_label_as_target=score_cfg["use_label_as_target"], eps=score_cfg["eps"],
#         # )
#         delta_map, _info, labels_used = compute_delta_map_gpu(
#         images_recon_all=recon_all,   # ← correct name from scoring.py

#         labels_fine=labels_fine,
#         # recon_images=recon_all,
#         pca_feats_recon=embed_result["pca_feats_recon"],
#         label_image=label_image,
#         label_pca_map=embed_result["label_pca_map"],
#         bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
#         smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
#         device=device,
#         gray_scale=gray_scale,
#         )
#     print(f"  Scored {len(labels_used)} superpixels [{algorithm}]")

#     # Stage 6: Evaluation
#     eval_cfg = cfg["eval"]
#     all_metrics = {}
#     for sigma in eval_cfg["delta_smooth_sigmas"]:
#         key = f"sigma_{sigma}" if sigma else "raw"
#         result = evaluate_delta_map(
#             delta_map=delta_map, labels_fine=labels_fine,
#             gt_mask_binary=gt_mask, anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
#             smooth_sigma=sigma,
#         )
#         all_metrics[key] = result
#         print(f"  [{key}] SP AUC: {result['sp_roc_auc']:.4f} | Pixel AUC: {result['px_roc_auc']:.4f}")

#     return all_metrics


# def _run_sweep(cfg: dict, args) -> None:
#     """Run evaluation on multiple samples and report averaged metrics."""
#     all_runs = {}
#     for sname in args.sample_names:
#         cfg_copy = deepcopy(cfg)
#         cfg_copy["data"]["sample_name"] = sname
#         metrics = _run_single_sample(cfg_copy)
#         all_runs[sname] = metrics

#     # Average across samples
#     first_keys = list(all_runs[args.sample_names[0]].keys())
#     print(f"\n{'='*60}")
#     print(f"SWEEP RESULTS ({len(args.sample_names)} images)")
#     print(f"Config: backbone={cfg['embeddings']['backbone']}, "
#           f"n_pca={cfg['pca']['n_components']}, "
#           f"bins_rgb={cfg['scoring']['bins_rgb']}, "
#           f"bins_pca={cfg['scoring']['bins_pca']}")
#     for metric_key in first_keys:
#         sp_aucs = [all_runs[s][metric_key]["sp_roc_auc"] for s in args.sample_names]
#         px_aucs = [all_runs[s][metric_key]["px_roc_auc"] for s in args.sample_names]
#         print(f"  [{metric_key}] Mean SP AUC: {np.mean(sp_aucs):.6f} | Mean Px AUC: {np.mean(px_aucs):.6f}")
#     print(f"{'='*60}")

#     # Save sweep results JSON
#     output_dir = cfg["eval"]["output_dir"]
#     os.makedirs(output_dir, exist_ok=True)
#     sweep_results = {
#         "config": {
#             "backbone": cfg["embeddings"]["backbone"],
#             "n_pca": cfg["pca"]["n_components"],
#             "bins_rgb": cfg["scoring"]["bins_rgb"],
#             "bins_pca": cfg["scoring"]["bins_pca"],
#         },
#         "sample_names": args.sample_names,
#         "per_sample": {},
#         "averaged": {},
#     }
#     for sname in args.sample_names:
#         sweep_results["per_sample"][sname] = {
#             k: {kk: vv for kk, vv in v.items() if kk != "curves"}
#             for k, v in all_runs[sname].items()
#         }
#     for metric_key in first_keys:
#         sp_aucs = [all_runs[s][metric_key]["sp_roc_auc"] for s in args.sample_names]
#         px_aucs = [all_runs[s][metric_key]["px_roc_auc"] for s in args.sample_names]
#         sweep_results["averaged"][metric_key] = {
#             "sp_roc_auc": float(np.mean(sp_aucs)),
#             "px_roc_auc": float(np.mean(px_aucs)),
#         }

#     sweep_name = (f"sweep_{cfg['embeddings']['backbone']}_pca{cfg['pca']['n_components']}"
#                   f"_rgb{cfg['scoring']['bins_rgb']}_pca{cfg['scoring']['bins_pca']}")
#     sweep_path = os.path.join(output_dir, f"{sweep_name}.json")
#     with open(sweep_path, "w") as f:
#         json.dump(sweep_results, f, indent=2, default=str)
#     print(f"Sweep results saved to {sweep_path}")


# def main():
#     parser = argparse.ArgumentParser(description="OOD Detection Evaluation Pipeline")
#     parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
#     parser.add_argument("--sample_name", type=str, default=None, help="Override sample name")
#     parser.add_argument("--skip_sampling", action="store_true", help="Skip DPS sampling, use existing results")
#     parser.add_argument("--no_plots", action="store_true",
#                         help="(no-op, retained for script compatibility) plots are never generated by this entry point")
#     parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/cuda:N)")
#     parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
#     parser.add_argument("--backbone", type=str, default=None,
#                         help="Override ResNet backbone (resnet18/50/101/152)")
#     parser.add_argument("--n_pca", type=int, default=None,
#                         help="Override number of PCA components")
#     parser.add_argument("--bins_rgb", type=int, default=None,
#                         help="Override RGB histogram bins")
#     parser.add_argument("--bins_pca", type=int, default=None,
#                         help="Override PCA histogram bins")
#     parser.add_argument("--sample_names", type=str, nargs="+", default=None,
#                         help="Run on multiple samples and average metrics")
#     parser.add_argument("--scorer", type=str, default="typical_set",
#                         choices=["typical_set", "local_gaussian"],
#                         help="Scoring algorithm: ood/scoring.py (typical_set) or ood/scoring_local_gaussian.py (rohan)")
#     parser.add_argument("--sigma_rohan", type=float, default=None,
#                         help="σ for local_gaussian scorer (in [0,1] image scale). "
#                              "If omitted, reads {results_dir}/sigma.txt when scorer=local_gaussian.")
#     args = parser.parse_args()

#     cfg = load_config(args.config)

#     # CLI overrides
#     if args.sample_name:
#         cfg["data"]["sample_name"] = args.sample_name
#     if args.skip_sampling:
#         cfg.setdefault("sampling", {})["enabled"] = False
#     if args.device:
#         cfg["embeddings"]["device"] = args.device
#     if args.output_dir:
#         cfg["eval"]["output_dir"] = args.output_dir
#     if args.backbone:
#         cfg["embeddings"]["backbone"] = args.backbone
#     if args.n_pca is not None:
#         cfg["pca"]["n_components"] = args.n_pca
#     if args.bins_rgb is not None:
#         cfg["scoring"]["bins_rgb"] = args.bins_rgb
#     if args.bins_pca is not None:
#         cfg["scoring"]["bins_pca"] = args.bins_pca
#     cfg.setdefault("scoring", {})["algorithm"] = args.scorer
#     if args.sigma_rohan is not None:
#         cfg["scoring"]["sigma_rohan"] = args.sigma_rohan

#     # Everything is routed through _run_sweep (works for 1 or many samples).
#     if not args.sample_names:
#         args.sample_names = [args.sample_name or cfg["data"]["sample_name"]]
#     _run_sweep(cfg, args)


# if __name__ == "__main__":
#     main()

### Commentoutfrom here
# #!/usr/bin/env python3
# """Evaluate OOD detection pipeline.

# Usage:
#     # PCA mode
#     python evaluate.py --config configs/experiment_ddad_native.yaml \\
#         --skip_sampling --scorer typical_set \\
#         --n_pca 5 --bins_pca 16 --sample_names samples_000 samples_001

#     # Autoencoder mode  (replaces PCA projection with AE latent space)
#     python evaluate.py --config configs/experiment_ddad_native.yaml \\
#         --skip_sampling --scorer typical_set \\
#         --autoencoder_path /path/to/ae.pth --n_pca 3 \\
#         --sample_names samples_000 samples_001

#     # Local Gaussian
#     python evaluate.py --config configs/experiment_ddad_native.yaml \\
#         --skip_sampling --scorer local_gaussian --sigma_rohan 0.1 \\
#         --sample_names samples_000 samples_001
# """

# import argparse
# import json
# import os
# import subprocess
# import sys
# from copy import deepcopy

# import numpy as np
# import torch
# import yaml

# from ood.data import (load_gt_mask, load_label_image, load_reconstructions,
#                       load_superpixel_mask, parse_sample_id)
# from ood.superpixels import recursive_subdivide
# from ood.embeddings import ResNetPixelEmbedder, embed_and_project
# from ood.scoring import compute_delta_map_gpu
# from ood.scoring_local_gaussian import compute_delta_map_local_gaussian
# from ood.metrics import evaluate_delta_map


# # ─────────────────────────────────────────────────────────────────────────────
# # PixelAutoEncoder  (same architecture as full_run_hist.py)
# # ─────────────────────────────────────────────────────────────────────────────

# class PixelAutoEncoder(torch.nn.Module):
#     def __init__(self, input_dim: int = 1417, latent_dim: int = 5):
#         super().__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Conv2d(input_dim, 512, 1, bias=False), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
#             torch.nn.Conv2d(512, 256, 1, bias=False),       torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
#             torch.nn.Conv2d(256, 64,  1, bias=False),       torch.nn.BatchNorm2d(64),  torch.nn.ReLU(True),
#             torch.nn.Conv2d(64, latent_dim, 1), torch.nn.Tanh(),
#         )
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Conv2d(latent_dim, 64,  1, bias=False), torch.nn.BatchNorm2d(64),  torch.nn.ReLU(True),
#             torch.nn.Conv2d(64,  256, 1, bias=False),        torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
#             torch.nn.Conv2d(256, 512, 1, bias=False),        torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
#             torch.nn.Conv2d(512, input_dim, 1),
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         return self.decoder(z), z


# def load_autoencoder(ae_path: str, embedder: ResNetPixelEmbedder,
#                      latent_dim: int, device: torch.device) -> PixelAutoEncoder:
#     with torch.no_grad():
#         embed_dim = embedder(torch.zeros(1, 3, 256, 256).to(device)).shape[1]
#     ae = PixelAutoEncoder(input_dim=embed_dim, latent_dim=latent_dim).to(device)
#     ae.load_state_dict(torch.load(ae_path, map_location=device))
#     ae.eval()
#     print(f"  AE loaded: {ae_path}  (embed_dim={embed_dim}, latent_dim={latent_dim})")
#     return ae


# def embed_with_ae(ae, embedder, label_image, images_recon_all, batch_size, device):
#     """AE-based projection — same interface as embed_and_project."""
#     from ood.embeddings import to_tensor01

#     with torch.no_grad():
#         label_feat = embedder(to_tensor01(label_image, device=str(device)))
#         _, label_z = ae(label_feat)
#         print(f"  Label z max abs: {label_z.abs().max().item():.4f}")
#     label_pca_map = label_z.squeeze(0).permute(1, 2, 0).cpu().numpy()   # (H, W, k)

#     x_all = torch.from_numpy(images_recon_all).float().permute(0, 3, 1, 2)
#     if x_all.max() > 1.0:
#         x_all = x_all / 255.0

#     pca_list = []
#     with torch.no_grad():
#         for i in range(0, x_all.shape[0], batch_size):
#             _, bz = ae(embedder(x_all[i:i+batch_size].to(device)))
#             pca_list.append(bz.cpu())
#     pca_feats_recon = torch.cat(pca_list, 0).permute(0, 2, 3, 1).numpy()  # (B, H, W, k)

#     return {"label_pca_map": label_pca_map, "pca_feats_recon": pca_feats_recon}


# # ─────────────────────────────────────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────────────────────────────────────

# def load_config(path: str) -> dict:
#     with open(path) as f:
#         return yaml.safe_load(f)


# # ─────────────────────────────────────────────────────────────────────────────
# # Single-sample evaluation
# # ─────────────────────────────────────────────────────────────────────────────

# def _run_single_sample(cfg: dict) -> dict:
#     data_cfg    = cfg["data"]
#     sample_name = data_cfg["sample_name"]
#     sample_id   = parse_sample_id(sample_name)
#     device      = torch.device(cfg["embeddings"].get("device", "cuda"))
#     figures_dir = os.path.join(data_cfg["figures_dir"], sample_name)
#     score_cfg   = cfg["scoring"]

#     print(f"\n=== OOD Evaluation: {sample_name} ===")

#     # Generate superpixel mask if missing
#     if not os.path.exists(os.path.join(figures_dir, "mask.png")):
#         input_image = os.path.join(data_cfg["image_dir"], f"{sample_id}.png")
#         os.makedirs(figures_dir, exist_ok=True)
#         if os.path.exists(input_image):
#             print(f"  Generating superpixels for {sample_name}...")
#             subprocess.run([sys.executable, "super_pixel_generation.py",
#                             f"--input_image={input_image}",
#                             f"--output_dir={figures_dir}"], check=True)

#     # ── Stage 1: Load data ────────────────────────────────────────────────
#     print("Loading data...")
#     recon_all   = load_reconstructions(
#         results_dir=data_cfg["results_dir"], sample_name=sample_name,
#         test_origin=data_cfg["test_origin"],
#         num_patches=cfg["sampling"].get("num_patches", 24),
#         bottom_suffix=data_cfg["bottom_suffix"],
#     )
#     label_image = load_label_image(
#         results_dir=data_cfg["results_dir"], sample_name=sample_name,
#         test_origin=data_cfg["test_origin"], bottom_suffix=data_cfg["bottom_suffix"],
#     )
#     sp_mask = load_superpixel_mask(figures_dir)
#     gt_mask = load_gt_mask(
#         path_template=data_cfg["gt_mask"]["path"],
#         sample=sample_id,
#         downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
#     )

#     # ── Stage 2: Superpixel refinement ───────────────────────────────────
#     sp_cfg = cfg["superpixels"]
#     labels_fine, final_ids, parent_map = recursive_subdivide(
#         img=label_image, labels=sp_mask,
#         var_threshold=sp_cfg["var_threshold"], min_pixels=sp_cfg["min_pixels"],
#         max_sub=sp_cfg["max_sub"], max_depth=sp_cfg["max_depth"],
#         compactness=sp_cfg["compactness"], alpha_grad=sp_cfg["alpha_grad"],
#         target_size=sp_cfg["target_size"],
#     )
#     print(f"  {len(np.unique(sp_mask))} -> {len(final_ids)} superpixels")

#     # ── Stage 3+4: Embed + project ────────────────────────────────────────
#     embed_cfg  = cfg["embeddings"]
#     latent_dim = cfg["pca"]["n_components"]
#     batch_size = embed_cfg.get("batch_size", 16)
#     ae_path    = score_cfg.get("autoencoder_path", "")

#     embedder = ResNetPixelEmbedder(
#         resnet_name=embed_cfg["backbone"],
#         layers=tuple(embed_cfg["layers"]),
#         use_patch_context=embed_cfg["use_patch_context"],
#         patchify_size=embed_cfg.get("patchify_size", 3),
#         proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer"),
#     ).to(device).eval()

#     if ae_path:
#         ae_model     = load_autoencoder(ae_path, embedder, latent_dim, device)
#         embed_result = embed_with_ae(ae_model, embedder, label_image, recon_all,
#                                      batch_size, device)
#         del ae_model
#         print(f"  Feature mode: AE  (latent_dim={latent_dim})")
#     else:
#         embed_result = embed_and_project(
#             embedder=embedder, label_image=label_image,
#             images_recon_all=recon_all, n_pca=latent_dim, device=str(device),
#         )
#         print(f"  Feature mode: PCA (n_components={latent_dim})")

#     del embedder
#     torch.cuda.empty_cache()

#     # ── Stage 5: Scoring ─────────────────────────────────────────────────
#     algorithm  = score_cfg.get("algorithm", "typical_set")
#     gray_scale = data_cfg.get("gray_scale", False)

#     if algorithm == "local_gaussian":
#         sigma_rohan = score_cfg.get("sigma_rohan")
#         if sigma_rohan is None:
#             sigma_path = os.path.join(data_cfg["results_dir"], "sigma.txt")
#             if os.path.exists(sigma_path):
#                 sigma_rohan = float(open(sigma_path).read().strip())
#                 print(f"  sigma_rohan={sigma_rohan} (from sigma.txt)")
#             else:
#                 raise ValueError("local_gaussian scorer needs --sigma_rohan")
#         delta_map, _info, labels_used = compute_delta_map_local_gaussian(
#             labels_fine=sp_mask,
#             parent_map={int(s): [int(s)] for s in np.unique(sp_mask)},
#             images_recon_all=recon_all, label_image=label_image,
#             sigma=sigma_rohan,
#             n_realizations=score_cfg.get("n_realizations", 1000),
#             min_pixels=score_cfg["min_pixels"],
#             metric=score_cfg.get("lg_metric", "typicality_unsigned"),
#         )
#     else:
#         delta_map, _info, labels_used = compute_delta_map_gpu(
#             labels_fine=labels_fine,
#             images_recon_all=recon_all,
#             pca_feats_recon=embed_result["pca_feats_recon"],
#             label_image=label_image,
#             label_pca_map=embed_result["label_pca_map"],
#             bins_rgb=score_cfg["bins_rgb"],
#             bins_pca=score_cfg["bins_pca"],
#             smooth_sigma=score_cfg["smooth_sigma"],
#             min_pixels=score_cfg["min_pixels"],
#             device=device,
#             gray_scale=gray_scale,
#         )

#     print(f"  Scored {len(labels_used)} superpixels [{algorithm}]")

#     # ── Stage 6: Evaluation ───────────────────────────────────────────────
#     eval_cfg    = cfg["eval"]
#     all_metrics = {}
#     for sigma in eval_cfg["delta_smooth_sigmas"]:
#         key    = f"sigma_{sigma}" if sigma else "raw"
#         result = evaluate_delta_map(
#             delta_map=delta_map, labels_fine=labels_fine,
#             gt_mask_binary=gt_mask,
#             anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
#             smooth_sigma=sigma,
#         )
#         all_metrics[key] = result
#         print(f"  [{key}] SP AUC={result['sp_roc_auc']:.4f}  "
#               f"Px AUC={result['px_roc_auc']:.4f}  "
#               f"SP AP={result['sp_ap']:.4f}  "
#               f"Px AP={result['px_ap']:.4f}")

#     return all_metrics


# # ─────────────────────────────────────────────────────────────────────────────
# # Sweep runner
# # ─────────────────────────────────────────────────────────────────────────────

# def _run_sweep(cfg: dict, args: argparse.Namespace) -> None:
#     all_runs = {}
#     for sname in args.sample_names:
#         cfg_copy                        = deepcopy(cfg)
#         cfg_copy["data"]["sample_name"] = sname
#         all_runs[sname]                 = _run_single_sample(cfg_copy)

#     first_keys = list(all_runs[args.sample_names[0]].keys())
#     ae_path    = cfg["scoring"].get("autoencoder_path", "")
#     mode       = f"AE({os.path.basename(ae_path)})" if ae_path else "PCA"

#     print(f"\n{'='*60}")
#     print(f"SWEEP RESULTS ({len(args.sample_names)} images)  [{mode}]")
#     print(f"n_pca={cfg['pca']['n_components']}  "
#           f"bins_rgb={cfg['scoring']['bins_rgb']}  "
#           f"bins_pca={cfg['scoring']['bins_pca']}  "
#           f"smooth_sigma={cfg['scoring']['smooth_sigma']}")

#     averaged = {}
#     for key in first_keys:
#         sp_aucs = [all_runs[s][key]["sp_roc_auc"] for s in args.sample_names]
#         px_aucs = [all_runs[s][key]["px_roc_auc"] for s in args.sample_names]
#         sp_aps  = [all_runs[s][key]["sp_ap"]       for s in args.sample_names]
#         px_aps  = [all_runs[s][key]["px_ap"]       for s in args.sample_names]
#         averaged[key] = {
#             "sp_roc_auc": float(np.mean(sp_aucs)),
#             "px_roc_auc": float(np.mean(px_aucs)),
#             "sp_ap":      float(np.mean(sp_aps)),
#             "px_ap":      float(np.mean(px_aps)),
#         }
#         print(f"  [{key}]  SP AUC={np.mean(sp_aucs):.4f}  "
#               f"Px AUC={np.mean(px_aucs):.4f}  "
#               f"SP AP={np.mean(sp_aps):.4f}  Px AP={np.mean(px_aps):.4f}")
#     print("=" * 60)

#     # Save JSON
#     output_dir = cfg["eval"]["output_dir"]
#     os.makedirs(output_dir, exist_ok=True)
#     ae_tag   = f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}_" if ae_path else "pca_"
#     run_name = (f"sweep_{ae_tag}{cfg['embeddings']['backbone']}"
#                 f"_n{cfg['pca']['n_components']}"
#                 f"_rgb{cfg['scoring']['bins_rgb']}"
#                 f"_pca{cfg['scoring']['bins_pca']}"
#                 f"_s{cfg['scoring']['smooth_sigma']}")
#     out_path = os.path.join(output_dir, f"{run_name}.json")
#     with open(out_path, "w") as f:
#         json.dump({
#             "config": {
#                 "backbone":        cfg["embeddings"]["backbone"],
#                 "n_pca":           cfg["pca"]["n_components"],
#                 "bins_rgb":        cfg["scoring"]["bins_rgb"],
#                 "bins_pca":        cfg["scoring"]["bins_pca"],
#                 "smooth_sigma":    cfg["scoring"]["smooth_sigma"],
#                 "autoencoder":     ae_path,
#             },
#             "sample_names": args.sample_names,
#             "per_sample":   {s: {k: {kk: vv for kk, vv in v.items() if kk != "curves"}
#                                  for k, v in all_runs[s].items()}
#                              for s in args.sample_names},
#             "averaged":     averaged,
#         }, f, indent=2, default=str)
#     print(f"Results saved: {out_path}")
#     return out_path, averaged


# # ─────────────────────────────────────────────────────────────────────────────
# # CLI
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--config",            required=True)
#     p.add_argument("--sample_name",       default=None)
#     p.add_argument("--sample_names",      nargs="+", default=None)
#     p.add_argument("--skip_sampling",     action="store_true")
#     p.add_argument("--no_plots",          action="store_true")
#     p.add_argument("--device",            default=None)
#     p.add_argument("--output_dir",        default=None)
#     p.add_argument("--backbone",          default=None)
#     p.add_argument("--n_pca",             type=int,   default=None,
#                    help="PCA components (PCA mode) or AE latent dim (AE mode)")
#     p.add_argument("--bins_rgb",          type=int,   default=None)
#     p.add_argument("--bins_pca",          type=int,   default=None)
#     p.add_argument("--smooth_sigma",      type=float, default=None)
#     p.add_argument("--scorer",            default="typical_set",
#                    choices=["typical_set", "local_gaussian"])
#     p.add_argument("--sigma_rohan",       type=float, default=None)
#     p.add_argument("--autoencoder_path",  default=None,
#                    help="Path to PixelAutoEncoder .pth checkpoint.  "
#                         "When set, replaces PCA projection with AE latent space.")
#     args = p.parse_args()

#     cfg = load_config(args.config)

#     if args.sample_name:              cfg["data"]["sample_name"]        = args.sample_name
#     if args.skip_sampling:            cfg.setdefault("sampling",{})["enabled"] = False
#     if args.device:                   cfg["embeddings"]["device"]        = args.device
#     if args.output_dir:               cfg["eval"]["output_dir"]          = args.output_dir
#     if args.backbone:                 cfg["embeddings"]["backbone"]      = args.backbone
#     if args.n_pca        is not None: cfg["pca"]["n_components"]         = args.n_pca
#     if args.bins_rgb     is not None: cfg["scoring"]["bins_rgb"]         = args.bins_rgb
#     if args.bins_pca     is not None: cfg["scoring"]["bins_pca"]         = args.bins_pca
#     if args.smooth_sigma is not None: cfg["scoring"]["smooth_sigma"]     = args.smooth_sigma
#     if args.sigma_rohan  is not None: cfg["scoring"]["sigma_rohan"]      = args.sigma_rohan
#     if args.autoencoder_path:         cfg["scoring"]["autoencoder_path"] = args.autoencoder_path
#     cfg.setdefault("scoring", {})["algorithm"] = args.scorer

#     if not args.sample_names:
#         args.sample_names = [args.sample_name or cfg["data"]["sample_name"]]

#     _run_sweep(cfg, args)


# if __name__ == "__main__":
#     main()
    

#!/usr/bin/env python3
"""Evaluate OOD detection — category/subcategory aware.

Directory convention:
    {results_root}_{category}_{subcategory}/
        samples_000/ ...
        samples_001/ ...

GT mask convention (tried in order):
    {gt_root}/{category}/{subcategory}/{sample_id}_mask.png
    {gt_root}/{subcategory}/{sample_id}_mask.png

Usage:
    # Single subcategory, specific samples
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --subcategory missing_wire \\
        --sample_names samples_000 samples_001

    # All subcategories, auto-discover all samples
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --all_subcategories \\
        --results_root ./results_patches_ddad_native

    # AE mode
    python evaluate.py --config configs/experiment_ddad_native.yaml \\
        --category cable --subcategory missing_wire \\
        --autoencoder_path /models/ae_cable_3.pth --n_pca 3
"""

import argparse, json, os, re, subprocess, sys
from copy import deepcopy
from glob import glob

import numpy as np
import torch
import yaml

from ood.data import (load_gt_mask, load_label_image, load_reconstructions,
                      load_superpixel_mask, parse_sample_id)
from ood.superpixels import recursive_subdivide
from ood.embeddings import ResNetPixelEmbedder, embed_and_project
from ood.scoring import compute_delta_map_gpu
from ood.scoring_local_gaussian import compute_delta_map_local_gaussian
from ood.metrics import evaluate_delta_map
import matplotlib.pyplot as plt


# ── PixelAutoEncoder ─────────────────────────────────────────────────────────
class PixelAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim=1417, latent_dim=5):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim,512,1,bias=False),torch.nn.BatchNorm2d(512),torch.nn.ReLU(True),
            torch.nn.Conv2d(512,256,1,bias=False),torch.nn.BatchNorm2d(256),torch.nn.ReLU(True),
            torch.nn.Conv2d(256,64,1,bias=False),torch.nn.BatchNorm2d(64),torch.nn.ReLU(True),
            torch.nn.Conv2d(64,latent_dim,1),torch.nn.Tanh())
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim,64,1,bias=False),torch.nn.BatchNorm2d(64),torch.nn.ReLU(True),
            torch.nn.Conv2d(64,256,1,bias=False),torch.nn.BatchNorm2d(256),torch.nn.ReLU(True),
            torch.nn.Conv2d(256,512,1,bias=False),torch.nn.BatchNorm2d(512),torch.nn.ReLU(True),
            torch.nn.Conv2d(512,input_dim,1))
    def forward(self,x):
        z=self.encoder(x); return self.decoder(z),z

def load_autoencoder(ae_path, embedder, latent_dim, device):
    with torch.no_grad():
        embed_dim = embedder(torch.zeros(1,3,256,256).to(device)).shape[1]
    ae = PixelAutoEncoder(input_dim=embed_dim, latent_dim=latent_dim).to(device)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.eval()
    print(f"  AE loaded: {ae_path}  (embed_dim={embed_dim}, latent_dim={latent_dim})")
    return ae

def embed_with_ae(ae, embedder, label_image, images_recon_all, batch_size, device):
    from ood.embeddings import to_tensor01
    with torch.no_grad():
        _, label_z = ae(embedder(to_tensor01(label_image, device=str(device))))
    label_pca_map = label_z.squeeze(0).permute(1,2,0).cpu().numpy()
    x_all = torch.from_numpy(images_recon_all).float().permute(0,3,1,2)
    if x_all.max() > 1.0: x_all = x_all / 255.0
    pca_list = []
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            _, bz = ae(embedder(x_all[i:i+batch_size].to(device)))
            pca_list.append(bz.cpu())
    return {"label_pca_map": label_pca_map,
            "pca_feats_recon": torch.cat(pca_list,0).permute(0,2,3,1).numpy()}


# ── Category / subcategory path helpers ──────────────────────────────────────

def resolve_results_dir(cfg, category, subcategory):
    d = cfg["data"]
    if "results_root" in d:
        return f"{d['results_root']}_{category}_{subcategory}"
    return d["results_dir"]

def resolve_gt_mask_template(cfg, category, subcategory):
    """Returns a path template containing {sample} for use with load_gt_mask."""
    d = cfg["data"]
    if "gt_root" in d:
        root = d["gt_root"]
        # Try most specific path first
        for cand in [
            os.path.join(root, category, subcategory, "{sample}_mask.png"),
            os.path.join(root, subcategory, "{sample}_mask.png"),
            os.path.join(root, "{sample}_mask.png"),
        ]:
            parent = os.path.dirname(cand.replace("{sample}", "PLACEHOLDER"))
            if os.path.isdir(parent):
                return cand
        # Return the most specific candidate even if directory not yet found
        return os.path.join(root, category, subcategory, "{sample}_mask.png")
    return d["gt_mask"]["path"]

def discover_subcategories(results_root, category):
    prefix  = f"{results_root}_{category}_"
    matches = sorted(glob(f"{prefix}*"))
    return [m[len(prefix):] for m in matches if os.path.isdir(m) and m[len(prefix):]]

def discover_samples(results_dir):
    if not os.path.isdir(results_dir):
        return []
    return sorted(e for e in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, e)) and e.startswith("samples_"))

def resolve_figures_dir(cfg, category, sample_name):
    d = cfg["data"]
    if "figures_root" in d:
        return os.path.join(d["figures_root"], category, sample_name)
    return os.path.join(d["figures_dir"], sample_name)


# ── Config ───────────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


# ── Single-sample evaluation ─────────────────────────────────────────────────
def _run_single_sample(cfg):
    data_cfg    = cfg["data"]
    sample_name = data_cfg["sample_name"]
    sample_id   = parse_sample_id(sample_name)
    device      = torch.device(cfg["embeddings"].get("device","cuda"))
    score_cfg   = cfg["scoring"]
    figures_dir = data_cfg.get("_figures_dir",
                               os.path.join(data_cfg["figures_dir"], sample_name))

    print(f"\n  [{sample_name}]")

    # Generate superpixel mask if missing
    if not os.path.exists(os.path.join(figures_dir,"mask.png")):
        inp = os.path.join(data_cfg.get("image_dir",""), f"{sample_id}.png")
        os.makedirs(figures_dir, exist_ok=True)
        if os.path.exists(inp):
            subprocess.run([sys.executable,"super_pixel_generation.py",
                            f"--input_image={inp}", f"--output_dir={figures_dir}"],check=True)

    # Load
    recon_all   = load_reconstructions(results_dir=data_cfg["results_dir"],
                      sample_name=sample_name, test_origin=data_cfg["test_origin"],
                      num_patches=cfg["sampling"].get("num_patches",24),
                      bottom_suffix=data_cfg["bottom_suffix"])
    label_image = load_label_image(results_dir=data_cfg["results_dir"],
                      sample_name=sample_name, test_origin=data_cfg["test_origin"],
                      bottom_suffix=data_cfg["bottom_suffix"])
    sp_mask     = load_superpixel_mask(figures_dir)
    gt_mask     = load_gt_mask(path_template=data_cfg["gt_mask"]["path"],
                               sample=sample_id,
                               downsample_factor=data_cfg["gt_mask"]["downsample_factor"])

    # category = data_cfg.get("category", getattr(cfg.get("eval", {}), "category", "cable"))
    # valid_mask = None
    # if category == "CT":
    #     import re
    #     import PIL.Image as Image
    #     m = re.search(r"samples_(?:image_)?(\d+)", sample_name)
    #     s_num = str(int(m.group(1))) if m else sample_id
        
    #     # Override gt_mask for CT
    #     gt_path = f"/data2/akheirandish3/id_new_warped_images/masks_ood_default_large/image_{s_num}_mask.png"
    #     if os.path.exists(gt_path):
    #         mask = np.array(Image.open(gt_path).convert("L"))[::2, ::2]  # fixed CT downsample=2
    #         gt_mask = (mask > 0).astype(np.uint8)
    #     else:
    #         print(f"  [WARN] CT GT mask not found at {gt_path}")
            
    #     # Load valid (body) mask for CT
    #     body_p = f"/data2/akheirandish3/id_new_warped_images/masks_body_ood_default_large/image_{s_num}_mask.png"
    #     if os.path.exists(body_p):
    #         body = np.array(Image.open(body_p).convert("L")) > 0
    #         valid_mask = body.astype(np.uint8)
    category = data_cfg.get("category", "cable")
    valid_mask = None

    if category == "CT":
        import re
        import PIL.Image as Image

        m = re.search(r"samples_(?:image_)?0*(\d+)", sample_name)
        if m is None:
            raise ValueError(f"Could not parse CT sample number from {sample_name}")

        s_int = str(int(m.group(1)))       # 92
        s_pad4 = f"{int(m.group(1)):04d}"  # 0092

        mask_dirs = [
            "/data2/akheirandish3/id_new_warped_images/masks_ood_med_med",
            "/data2/akheirandish3/id_new_warped_images/masks_ood_default_large",
        ]

        candidates = []
        for md in mask_dirs:
            candidates.extend([
                os.path.join(md, f"image_{s_pad4}_mask.png"),
                os.path.join(md, f"image_{s_int}_mask.png"),
                os.path.join(md, f"{s_pad4}_mask.png"),
                os.path.join(md, f"{s_int}_mask.png"),
            ])

        gt_path = next((p for p in candidates if os.path.exists(p)), None)

        if gt_path is None:
            print("Tried these CT GT mask paths:")
            for p in candidates:
                print("  ", p)
            raise FileNotFoundError(f"CT GT mask not found for {sample_name}")

        print(f"  CT GT mask: {gt_path}")
        mask = np.array(Image.open(gt_path).convert("L"))[::2, ::2]
        gt_mask = (mask > 0).astype(np.uint8)

        body_candidates = [
            f"/data2/akheirandish3/id_new_warped_images/masks_body_ood_default_large/image_{s_pad4}_mask.png",
            f"/data2/akheirandish3/id_new_warped_images/masks_body_ood_default_large/image_{s_int}_mask.png",
        ]
        body_p = next((p for p in body_candidates if os.path.exists(p)), None)
        if body_p is not None:
            body = np.array(Image.open(body_p).convert("L")) > 0
            valid_mask = body.astype(np.uint8)

    else:
        gt_mask = load_gt_mask(
            path_template=data_cfg["gt_mask"]["path"],
            sample=sample_id,
            downsample_factor=data_cfg["gt_mask"]["downsample_factor"],
        )
    # Superpixel refinement
    sp_cfg = cfg["superpixels"]
    labels_fine, final_ids, parent_map = recursive_subdivide(
        img=label_image, labels=sp_mask,
        var_threshold=sp_cfg["var_threshold"], min_pixels=sp_cfg["min_pixels"],
        max_sub=sp_cfg["max_sub"], max_depth=sp_cfg["max_depth"],
        compactness=sp_cfg["compactness"], alpha_grad=sp_cfg["alpha_grad"],
        target_size=sp_cfg["target_size"])
    print(f"  {len(np.unique(sp_mask))} -> {len(final_ids)} superpixels")
    _plot_superpixels(label_image, labels_fine, sp_mask,
                    save_path=os.path.join(figures_dir, f"{sample_name}_superpixels.png"))
    # Embed
    embed_cfg  = cfg["embeddings"]
    latent_dim = cfg["pca"]["n_components"]
    ae_path    = score_cfg.get("autoencoder_path","")
    embedder   = ResNetPixelEmbedder(
        resnet_name=embed_cfg["backbone"], layers=tuple(embed_cfg["layers"]),
        use_patch_context=embed_cfg["use_patch_context"],
        patchify_size=embed_cfg.get("patchify_size",3),
        proj_dim_per_layer=embed_cfg.get("proj_dim_per_layer")).to(device).eval()

    if ae_path:
        ae_model     = load_autoencoder(ae_path, embedder, latent_dim, device)
        embed_result = embed_with_ae(ae_model, embedder, label_image, recon_all,
                                     embed_cfg.get("batch_size",16), device)
        del ae_model
    else:
        embed_result = embed_and_project(embedder=embedder, label_image=label_image,
            images_recon_all=recon_all, n_pca=latent_dim, device=str(device))
    del embedder; torch.cuda.empty_cache()

    # Score
    algorithm  = score_cfg.get("algorithm","typical_set")
    gray_scale = data_cfg.get("gray_scale", False)
    if algorithm == "local_gaussian":
        sigma_r = score_cfg.get("sigma_rohan")
        if sigma_r is None:
            sp = os.path.join(data_cfg["results_dir"],"sigma.txt")
            sigma_r = float(open(sp).read().strip()) if os.path.exists(sp) else None
        if sigma_r is None: raise ValueError("local_gaussian scorer needs --sigma_rohan")
        delta_map,_,labels_used = compute_delta_map_local_gaussian(
            labels_fine=sp_mask,
            parent_map={int(s):[int(s)] for s in np.unique(sp_mask)},
            images_recon_all=recon_all, label_image=label_image,
            sigma=sigma_r, n_realizations=score_cfg.get("n_realizations",1000),
            min_pixels=score_cfg["min_pixels"], metric=score_cfg.get("lg_metric","typicality_unsigned"))
    else:
        delta_map,_,labels_used = compute_delta_map_gpu(
            labels_fine=labels_fine, images_recon_all=recon_all,
            pca_feats_recon=embed_result["pca_feats_recon"],
            label_image=label_image, label_pca_map=embed_result["label_pca_map"],
            bins_rgb=score_cfg["bins_rgb"], bins_pca=score_cfg["bins_pca"],
            smooth_sigma=score_cfg["smooth_sigma"], min_pixels=score_cfg["min_pixels"],
            device=device, gray_scale=gray_scale)
    print(f"  Scored {len(labels_used)} SPs [{algorithm}]")

    # ── Shape reconciliation ──────────────────────────────────────────────
    # delta_map shape is driven by recon image size; gt_mask shape by
    # downsample_factor.  For some categories (e.g. capsule) these differ.
    if gt_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] Shape mismatch: delta_map={delta_map.shape} "
              f"gt_mask={gt_mask.shape} — resizing gt_mask to match delta_map.")
        gt_mask = _resize(
            gt_mask.astype(np.float32),
            output_shape=delta_map.shape,
            order=0,              # nearest-neighbour — preserves binary values
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)

    if labels_fine.shape != delta_map.shape:
        print(f"  [WARN] labels_fine shape {labels_fine.shape} != "
              f"delta_map shape {delta_map.shape} — resizing labels_fine.")
        from skimage.transform import resize as _resize
        labels_fine = _resize(
            labels_fine.astype(np.float32),
            output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.int32)
        
    if valid_mask is not None and valid_mask.shape != delta_map.shape:
        from skimage.transform import resize as _resize
        print(f"  [WARN] valid_mask shape mismatch, resizing to {delta_map.shape}.")
        valid_mask = _resize(
            valid_mask.astype(np.float32),
            output_shape=delta_map.shape,
            order=0, mode="edge", preserve_range=True, anti_aliasing=False,
        ).astype(np.uint8)

    # Evaluate
    eval_cfg = cfg["eval"]
    all_metrics = {}
    for sigma in eval_cfg["delta_smooth_sigmas"]:
        key    = f"sigma_{sigma}" if sigma else "raw"
        result = evaluate_delta_map(delta_map=delta_map, labels_fine=labels_fine,
            gt_mask_binary=gt_mask, anomaly_threshold=eval_cfg["sp_anomaly_threshold"],
            smooth_sigma=sigma, valid_mask=valid_mask)
        all_metrics[key] = result
        print(f"  [{key}] SP={result['sp_roc_auc']:.4f} Px={result['px_roc_auc']:.4f} "
              f"SP_AP={result['sp_ap']:.4f} Px_AP={result['px_ap']:.4f} SNR={result['snr']:.2f}")
    return all_metrics


# ── Subcategory runner ────────────────────────────────────────────────────────
def _run_subcategory(base_cfg, category, subcategory, sample_names, output_dir):
    print(f"\n{'═'*60}")
    print(f"  {category} / {subcategory}")
    print(f"{'═'*60}")

    cfg = deepcopy(base_cfg)
    results_dir = resolve_results_dir(base_cfg, category, subcategory)
    gt_template = resolve_gt_mask_template(base_cfg, category, subcategory)

    cfg["data"]["results_dir"]     = results_dir
    # cfg["data"]["gt_mask"]["path"] = gt_template
    cfg["data"]["category"] = category
    cfg["data"]["subcategory"] = subcategory
    cfg["data"]["figures_dir"] = "./figures_CT" if category == "CT" else os.path.join(base_cfg["data"]["figures_dir"], category, subcategory)

    if category != "CT":
        cfg["data"]["gt_mask"]["path"] = gt_template

    # Auto-discover samples if not given
    names = sample_names or discover_samples(results_dir)
    if not names:
        print(f"  [WARN] No samples found in {results_dir}")
        return {}
    print(f"  Samples ({len(names)}): {names[:5]}{'...' if len(names)>5 else ''}")
    print(f"  Results dir : {results_dir}")
    print(f"  GT template : {gt_template}")

    results, failed = {}, []
    for sname in names:
        c = deepcopy(cfg)
        c["data"]["sample_name"] = sname
        c["data"]["_figures_dir"] = resolve_figures_dir(base_cfg, category, sname)
        try:
            results[sname] = _run_single_sample(c)
        except Exception as exc:
            print(f"  [ERROR] {sname}: {exc}")
            failed.append(sname)

    # Average & save
    first_keys = list(next(iter(results.values())).keys()) if results else []
    averaged = {}
    if results:
        print(f"\n  AVERAGES — {subcategory} ({len(results)} samples):")
        for key in first_keys:
            vals = [results[s][key] for s in results if key in results[s]]
            averaged[key] = {
                "sp_roc_auc": float(np.mean([v["sp_roc_auc"] for v in vals])),
                "px_roc_auc": float(np.mean([v["px_roc_auc"] for v in vals])),
                "sp_ap":      float(np.mean([v["sp_ap"]       for v in vals])),
                "px_ap":      float(np.mean([v["px_ap"]       for v in vals])),
                "snr":        float(np.nanmean([v.get("snr", np.nan) for v in vals])),
            }
            a = averaged[key]
            print(f"    [{key}] SP={a['sp_roc_auc']:.4f} Px={a['px_roc_auc']:.4f} "
                  f"SP_AP={a['sp_ap']:.4f} Px_AP={a['px_ap']:.4f} SNR={a['snr']:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    ae_path = base_cfg["scoring"].get("autoencoder_path","")
    ae_tag  = f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}_" if ae_path else ""
    jpath   = os.path.join(output_dir, f"{ae_tag}{category}_{subcategory}.json")
    with open(jpath,"w") as f:
        json.dump({"category":category,"subcategory":subcategory,
                   "results_dir":results_dir,"gt_template":gt_template,
                   "samples":list(results),"failed":failed,
                   "per_sample":{s:{k:{kk:vv for kk,vv in v.items() if kk!="curves"}
                                    for k,v in results[s].items()} for s in results},
                   "averaged":averaged}, f, indent=2, default=str)
    print(f"  Saved: {jpath}")
    return results


# ── Category summary ──────────────────────────────────────────────────────────
def _summarize_category(all_subcat_results, category, output_dir, base_cfg):
    print(f"\n{'═'*60}")
    print(f"CATEGORY SUMMARY: {category}  ({len(all_subcat_results)} subcategories)")
    print(f"{'═'*60}")
    rows = []
    for subcat, sample_results in all_subcat_results.items():
        if not sample_results: continue
        first_keys = list(next(iter(sample_results.values())).keys())
        for key in first_keys:
            vals = [sample_results[s][key] for s in sample_results if key in sample_results[s]]
            if not vals: continue
            rows.append({"subcategory":subcat,"eval_sigma":key,"n":len(sample_results),
                "sp_roc_auc":float(np.mean([v["sp_roc_auc"] for v in vals])),
                "px_roc_auc":float(np.mean([v["px_roc_auc"] for v in vals])),
                "sp_ap":float(np.mean([v["sp_ap"] for v in vals])),
                "px_ap":float(np.mean([v["px_ap"] for v in vals])),
                "snr":float(np.nanmean([v.get("snr", np.nan) for v in vals]))})
            r = rows[-1]
            print(f"  {subcat:<22} [{key}]  SP={r['sp_roc_auc']:.4f}  "
                  f"Px={r['px_roc_auc']:.4f}  SP_AP={r['sp_ap']:.4f}  "
                  f"Px_AP={r['px_ap']:.4f}  SNR={r['snr']:.2f} (n={r['n']})")
    if rows:
        print(f"\n  OVERALL  SP={np.mean([r['sp_roc_auc'] for r in rows]):.4f}  "
              f"Px={np.mean([r['px_roc_auc'] for r in rows]):.4f} SNR={np.nanmean([float(r.get('snr', np.nan)) for r in rows]):.2f}")
    ae_path = base_cfg["scoring"].get("autoencoder_path","")
    ae_tag  = f"ae_{os.path.splitext(os.path.basename(ae_path))[0]}_" if ae_path else ""
    jpath   = os.path.join(output_dir, f"{ae_tag}{category}_summary.json")
    with open(jpath,"w") as f:
        json.dump({"category":category,"rows":rows}, f, indent=2, default=str)
    print(f"Summary saved: {jpath}")

def _plot_superpixels(
    img:          np.ndarray,    # (H, W, 3) uint8 label image
    labels_fine:  np.ndarray,    # (H, W) int refined superpixel map
    labels_coarse: np.ndarray,   # (H, W) int original coarse SP mask
    save_path:    str = None,
    show:         bool = False,
) -> None:
    """
    Plot 4 panels:
      1. Original image
      2. Coarse superpixel boundaries (from mask.png)
      3. Refined superpixel boundaries (after recursive_subdivide)
      4. Refined SPs coloured by region mean
    """
    from skimage.segmentation import find_boundaries
    from skimage.util import img_as_float
    from skimage.color import label2rgb

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    def _overlay(ax, image, labels, title, color="yellow"):
        vis = img_as_float(image).copy()
        b   = find_boundaries(labels, mode="thick")
        if vis.ndim == 2:
            vis = np.repeat(vis[..., None], 3, axis=2)
        vis[b, :] = {
            "yellow": [1.0, 1.0, 0.0],
            "red":    [1.0, 0.2, 0.2],
            "cyan":   [0.0, 1.0, 1.0],
        }.get(color, [1.0, 1.0, 0.0])
        ax.imshow(vis)
        ax.set_title(f"{title}\n({len(np.unique(labels))} regions)")
        ax.axis("off")

    # Panel 1 — raw image
    axes[0].imshow(img)
    axes[0].set_title("Label image"); axes[0].axis("off")

    # Panel 2 — coarse SP boundaries
    _overlay(axes[1], img, labels_coarse, "Coarse SPs (mask.png)", color="cyan")

    # Panel 3 — refined SP boundaries
    _overlay(axes[2], img, labels_fine,   "Refined SPs", color="yellow")

    # Panel 4 — each SP coloured by its mean RGB
    colored = label2rgb(labels_fine, image=img, kind="avg", bg_label=-1)
    b_fine  = find_boundaries(labels_fine, mode="thick")
    colored[b_fine, :] = 0.0          # black boundary lines
    axes[3].imshow(colored)
    axes[3].set_title(f"SP mean colours\n({len(np.unique(labels_fine))} regions)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved superpixel plot: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True)
    p.add_argument("--category",          default=None,
                   help="Dataset category, e.g. cable")
    p.add_argument("--subcategory",       default=None,
                   help="Single defect subcategory, e.g. missing_wire")
    p.add_argument("--all_subcategories", action="store_true",
                   help="Auto-discover and run ALL subcategories for --category")
    p.add_argument("--results_root",      default=None,
                   help="Prefix: actual dir = {results_root}_{category}_{subcategory}")
    p.add_argument("--gt_root",           default=None,
                   help="GT mask root. Looks for {gt_root}/{category}/{subcategory}/{sample}_mask.png")
    p.add_argument("--sample_name",       default=None)
    p.add_argument("--sample_names",      nargs="+", default=None,
                   help="Explicit samples. If omitted, all samples in results_dir are used.")
    p.add_argument("--skip_sampling",     action="store_true")
    p.add_argument("--no_plots",          action="store_true")
    p.add_argument("--device",            default=None)
    p.add_argument("--output_dir",        default=None)
    p.add_argument("--backbone",          default=None)
    p.add_argument("--n_pca",             type=int,   default=None)
    p.add_argument("--bins_rgb",          type=int,   default=None)
    p.add_argument("--bins_pca",          type=int,   default=None)
    p.add_argument("--smooth_sigma",      type=float, default=None)
    p.add_argument("--scorer",            default="typical_set",
                   choices=["typical_set","local_gaussian"])
    p.add_argument("--sigma_rohan",       type=float, default=None)
    p.add_argument("--autoencoder_path",  default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("sampling",{})["enabled"] = False
    cfg.setdefault("scoring",{})["algorithm"] = args.scorer

    if args.device:                   cfg["embeddings"]["device"]              = args.device
    if args.output_dir:               cfg["eval"]["output_dir"]                = args.output_dir
    if args.backbone:                 cfg["embeddings"]["backbone"]            = args.backbone
    if args.n_pca        is not None: cfg["pca"]["n_components"]               = args.n_pca
    if args.bins_rgb     is not None: cfg["scoring"]["bins_rgb"]               = args.bins_rgb
    if args.bins_pca     is not None: cfg["scoring"]["bins_pca"]               = args.bins_pca
    if args.smooth_sigma is not None: cfg["scoring"]["smooth_sigma"]           = args.smooth_sigma
    if args.sigma_rohan  is not None: cfg["scoring"]["sigma_rohan"]            = args.sigma_rohan
    if args.autoencoder_path:         cfg["scoring"]["autoencoder_path"]       = args.autoencoder_path
    if args.results_root:             cfg["data"]["results_root"]              = args.results_root
    if args.gt_root:                  cfg["data"]["gt_root"]                   = args.gt_root

    output_dir = cfg["eval"]["output_dir"]
    category   = args.category or cfg["data"].get("category","cable")
    sample_names = args.sample_names or ([args.sample_name] if args.sample_name else None)

    if args.all_subcategories:
        rroot   = cfg["data"].get("results_root", cfg["data"].get("results_dir",""))
        subcats = discover_subcategories(rroot, category)
        if not subcats:
            print(f"[ERROR] No subcategories found for {rroot}_{category}_*")
            sys.exit(1)
        print(f"Discovered subcategories: {subcats}")
    elif args.subcategory:
        subcats = [args.subcategory]
    else:
        subcats = [None]

    all_subcat_results = {}
    for subcat in subcats:
        if subcat is None:
            # No subcategory — plain mode
            names = sample_names or [cfg["data"].get("sample_name","samples_000")]
            for sname in names:
                c = deepcopy(cfg)
                c["data"]["sample_name"] = sname
                c["data"]["_figures_dir"] = os.path.join(cfg["data"]["figures_dir"], sname)
                _run_single_sample(c)
        else:
            all_subcat_results[subcat] = _run_subcategory(
                cfg, category, subcat, sample_names, output_dir)

    if len(all_subcat_results) > 1:
        _summarize_category(all_subcat_results, category, output_dir, cfg)

if __name__ == "__main__":
    main()