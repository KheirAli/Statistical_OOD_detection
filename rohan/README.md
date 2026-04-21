# OOD Detection via Local Gaussianity

Out-of-distribution detection for diffusion-based image inpainting using a local Gaussian approximation of the posterior.

## Files

- `algorithm.tex` -- Derivation of the algorithm: models the posterior as a local Gaussian, recovers the precision matrix from the denoised-sample covariance via eigendecomposition, and defines the per-superpixel OOD metric.
- `theory_algorithm_rohan.ipynb` -- Implementation. Loads reconstructed images, estimates the covariance in superpixel space, solves for the precision matrix, and computes all metric combinations (pixel mode, metric type, aggregation, channel reduction).

## Data

Sample reconstructions for `image_name="terminator"` (N=20 seeds × 6 sigmas in
{0.125, 0.25, 0.375, 0.5, 0.75, 1.0}) are **no longer tracked in the repo** to
keep it lean. They live outside the repo tree, e.g. at:

```
/home/rohan/ood/rohan_reference_data/terminator/
  label/00000.png
  superpixels/labels_1000.png
  recon/sigma{s}/{seed}_00000.png
```

The notebook's first cell builds `data_dir` from the current working directory
+ `"data"`. To run it, either:

1. Symlink the reference data back in: `ln -s /home/rohan/ood/rohan_reference_data rohan/data`
2. Or edit the notebook's `data_dir` to point at the external location.

Ported production-ready implementation lives at `../ood/scoring_local_gaussian.py`.
