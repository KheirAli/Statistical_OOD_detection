# OOD Detection via Local Gaussianity

Out-of-distribution detection for diffusion-based image inpainting using a local Gaussian approximation of the posterior.

## Files

- `algorithm.tex` -- Derivation of the algorithm: models the posterior as a local Gaussian, recovers the precision matrix from the denoised-sample covariance via eigendecomposition, and defines the per-superpixel OOD metric.
- `theory_algorithm_rohan.ipynb` -- Implementation. Loads reconstructed images, estimates the covariance in superpixel space, solves for the precision matrix, and computes all metric combinations (pixel mode, metric type, aggregation, channel reduction).

## Data

`data/terminator/` contains the images needed to run the notebook for `image_name="terminator"`, `N=20` seeds, and `sigma` in {0.125, 0.25, 0.375, 0.5, 0.75, 1.0}.

```
data/terminator/
  label/00000.png              Ground-truth image (shared across all sigmas)
  superpixels/labels_1000.png  Superpixel segmentation mask (1000 superpixels)
  recon/sigma{s}/{seed}_00000.png   Denoised reconstructions (6 sigmas x 20 seeds)
```

To run with a different sigma, change the `sigma` parameter at the top of the notebook.
