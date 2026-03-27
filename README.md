# Statistical Out-of-Distribution Detection

This repository contains code for statistical out-of-distribution (OOD) detection methods. The implementation is built on top of the **Diffusion Posterior Sampling** framework.

## Overview

Out-of-distribution detection is a critical task in machine learning that identifies when a model encounters data that differs significantly from its training distribution. This repository provides statistical methods for OOD detection using diffusion-based approaches.

## Prerequisites

- Python 3.6+
- PyTorch
- CUDA-compatible GPU (recommended)
- Additional dependencies as specified in `requirements.txt` (if available)

## Installation

### Step 1: Clone the Diffusion Posterior Sampling Repository

First, you need to clone the base Diffusion Posterior Sampling repository:

```bash
git clone https://github.com/[diffusion-posterior-sampling-repo-url]
cd diffusion-posterior-sampling
```

> **Note**: Replace `[diffusion-posterior-sampling-repo-url]` with the actual URL of the Diffusion Posterior Sampling repository.

### Step 2: Clone This Repository

Clone this Statistical OOD Detection repository:

```bash
git clone https://github.com/KheirAli/Statistical_OOD_detection.git
```

### Step 3: Replace Files

Copy and replace the files from this repository into the Diffusion Posterior Sampling directory:

```bash
cp -r Statistical_OOD_detection/* diffusion-posterior-sampling/
```

This will overwrite the necessary files in the base repository with the OOD detection implementations.

### Step 4: Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── [modified_files]          # Files that replace those in DPS repo
└── [additional_scripts]      # Additional OOD detection scripts
```

## Usage

### Basic Usage

[Add specific usage instructions here once files are examined]

```python
# Example usage
python main.py --config configs/ood_detection.yaml
```

### Configuration

Modify the configuration files to adjust:
- Detection thresholds
- Model parameters
- Dataset paths
- Evaluation metrics

## Methodology

This implementation leverages diffusion models for statistical OOD detection by:

1. **Posterior Sampling**: Using diffusion posterior sampling to analyze data distributions
2. **Statistical Testing**: Applying statistical methods to identify distribution shifts
3. **Threshold-based Detection**: Implementing adaptive thresholds for OOD classification

## Datasets

Supported datasets include:
- CIFAR-10
- CIFAR-100
- ImageNet
- [Add other supported datasets]

### Preparing Datasets

[Add dataset preparation instructions]

```bash
# Example dataset download/preparation
python prepare_data.py --dataset cifar10
```

## Evaluation

To evaluate OOD detection performance:

```bash
python evaluate.py --model [model_path] --ood_dataset [dataset_name]
```

### Metrics

The following metrics are used for evaluation:
- **AUROC** (Area Under Receiver Operating Characteristic)
- **AUPR** (Area Under Precision-Recall curve)
- **FPR95** (False Positive Rate at 95% True Positive Rate)
- **Detection Accuracy**

## Results

[Add benchmark results and performance comparisons]

| Method | CIFAR-10 vs SVHN | CIFAR-10 vs Places365 | CIFAR-100 vs CIFAR-10 |
|--------|------------------|----------------------|----------------------|
| Baseline | XX.X% | XX.X% | XX.X% |
| Statistical OOD | XX.X% | XX.X% | XX.X% |

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{statistical_ood_detection,
  author = {[Your Name]},
  title = {Statistical Out-of-Distribution Detection},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/KheirAli/Statistical_OOD_detection}
}
```

Also cite the base Diffusion Posterior Sampling work:

```bibtex
[Add DPS citation here]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Acknowledgments

- Base implementation from Diffusion Posterior Sampling
- [Add other acknowledgments]

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Troubleshooting

### Common Issues

**Issue 1: Import errors after file replacement**
- Solution: Ensure all dependencies are installed with `pip install -r requirements.txt`

**Issue 2: CUDA out of memory**
- Solution: Reduce batch size in the configuration file

**Issue 3: File path errors**
- Solution: Verify that all files from this repo are correctly placed in the DPS directory

## References

- [Link to relevant papers]
- [Link to Diffusion Posterior Sampling paper/repo]
- [Other relevant references]

---

**Last Updated**: March 2026
