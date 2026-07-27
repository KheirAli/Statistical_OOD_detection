# Metrics: definitions, code, and provenance

Code + exact definitions behind every column of the all-method comparison
tables (MVTec-15 / CT / xray-50 / faces). All metrics operate on per-image
anomaly heatmaps smoothed with a Gaussian of sigma=5 (the project's reference
operating point) unless stated otherwise.

## Where each metric is implemented

| Metric | Implementation | Notes |
|---|---|---|
| px ROC-AUC (per-image) | `ood/metrics.py` (`manual_roc_curve`/`manual_auc`, used by `evaluate_delta_map`) | reference metric; per-image, then subcat/category mean |
| px AP (per-image) | `ood/metrics.py::manual_average_precision` | validated bit-exact vs recorded baseline values |
| px SNR | `ood/metrics.py` (`evaluate_delta_map`, ~line 309) | **z-score**: `(mean_OOD - mean_ID) / std_ID` on the smoothed map. NOT a ratio of means — paper text at l.285-286 should be fixed accordingly |
| px SNR (paper) | `ood/metrics.py` (`snr_paper` key), `extra_map_metrics.py::px_snr_paper` | the paper's ratio definition `mean_OOD / mean_ID`, computed on the min-max normalized map (keeps both means non-negative); all recorded table SNR values are the z-score, not this |
| mask-MSE | `ood/metrics.py` (`evaluate_delta_map`) | min-max normalize map to [0,1] over the valid region, MSE vs binary GT. **Stored under the JSON key `psnr`** (historical misnomer) |
| PSNR (dB) | derived | `mean_i 10*log10(1 / mask-MSE_i)` |
| AUPRO | `tools/metrics/anomalib_mvtec_ct_metrics.py::aupro`, `anomalib_xray50_metrics.py::aupro` | per-region overlap AUC, 8-connected GT components, global FPR <= 0.3, normalized |
| pooled AUROC / AUPR / F1Max | same two scripts (`dataset_metrics` / `evaluate`) | pixels concatenated across images (per category for MVTec), one global threshold; F1Max over an 8192-bin histogram |
| per-image F1Max | same scripts | oracle threshold per image, then mean |
| per-image AUROC | same scripts | sanity anchor; reproduces the recorded per-image means exactly |

## Scripts

- `anomalib_mvtec_ct_metrics.py` — AUPRO + pooled metrics for MVTec (15 cats)
  and CT (body-masked), for ours / PatchCore / SimpleNet / SuperSimpleNet /
  DDAD-CT. Run: `python tools/metrics/anomalib_mvtec_ct_metrics.py {mvtec|ct}`.
  (Working copy also at `results_eval_anomalib_metrics/mvtec_ct_metrics.py`.)
- `anomalib_xray50_metrics.py` — same metric family for the 50-image xray
  scissors set (32 b1 + 18 b2), incl. DDAD saved x50 maps and our delta-map
  dumps. (Working copy at `results_eval_xray_baselines_b2/anomalib_metrics.py`.)
- `extra_map_metrics.py` — per-image px AP / SNR / mask-MSE / PSNR from saved
  maps, for the cells not present in any recorded metrics.json: DDAD on
  MVTec/CT/faces (+ xray SNR), GLASS mask-MSE, ours from delta-map dumps, and
  a from-heatmap recompute path for the feature baselines. See its docstring
  for targets and map locations.

## Our delta-map dumps

`evaluate.py` writes per-sample npz dumps (`delta_map` raw float32, `gt_mask`,
`valid_mask`, everything at 256x256) when the env var `DUMP_DELTA_DIR` is set,
e.g.

```bash
DUMP_DELTA_DIR=results_eval_anomalib_metrics/ours_ct_maps_released \
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/experiment_ddad_native_xray.yaml \
  --category CT --all_subcategories --results_root ./results_patches_ddad_native \
  --gt_root /data/akheirandish3/mvtec_ad/CT/ground_truth --skip_sampling \
  --autoencoder_path models/pixel_autoencoder_CT.pth \
  --n_pca 3 --bins_pca 64 --bins_rgb 64 --sigma_rohan 1.0 \
  --superpixel_target_size 30 --smooth_sigma 1 --max_reconstructions 40 \
  --no_plots --output_dir ./results_eval_CT_released_mapdump
```

Dumps are raw (unsmoothed): apply `gaussian_filter(map, 5.0)` before scoring.
Existing dump sets under `results_eval_anomalib_metrics/`:
`ours_mvtec_maps{,_s3}/`, `ours_ct_maps/` (seeded AE), `ours_ct_maps_released/`
(released CT AE), `ours_faces_maps/`; xray dumps under
`results_eval_xray_baselines_b2/ours_maps/`.

## Protocol details

- **Aggregation**: MVTec = per-image -> defect-subcat/category mean ->
  unweighted mean over 15 categories (`/good/` and `combined` excluded).
  CT/xray/faces = plain mean over images.
- **Valid region**: CT restricts every metric to the per-image body mask
  (`masks_body_ood_default_large`); other datasets use all pixels.
- **CT ground truth**: `masks_ood_med_med/image_XXXX_mask.png`, loaded
  grayscale, `[::2, ::2]` downsample, binarized `> 0` (hardcoded in
  `evaluate.py`'s CT branch — the `--gt_root` flag is ignored for CT).
- **No normal test images** in any of our sets -> image-level metrics
  (image AUROC, PIMO/AUPIMO) are not computable.
- **Known gotchas**: the harness JSON `psnr` field stores mask-MSE, not dB;
  GLASS's near-zero (sparse) maps make its mask-MSE/PSNR look best while its
  AUC/AP are mid-pack — treat the mask-MSE family as a map-cleanliness
  measure, not a detection measure.

## Map sources for other methods

- DDAD: `DDAD/hparam_sweep/mvtec_maps/{reference,v7}_<cat>.pt`,
  `ct_maps_reference.pt`, `faces_maps.pt`, `mvtec_maps/facesv7_faces.pt`,
  `results_eval_xray_baselines_b2/ddad_x50_{reference,v7}_maps.pt`
  (dumped via `DDAD_SAVE_MAPS` reruns; each validated to reproduce the
  recorded per-image AUROC exactly).
- PatchCore / SimpleNet / SuperSimpleNet / GLASS:
  `/data2/rohan/baseline_eval_results_v2/<ds>/<method>/heatmaps/*_amap_smooth.npy`
  (already sigma-5 smoothed) + `metrics.json` with recorded per-image
  px_auc/px_ap/px_snr.
