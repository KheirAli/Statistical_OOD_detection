# Statistical Out-of-Distribution Detection

This repository contains code for statistical out-of-distribution (OOD) detection methods. The implementation is built on top of the **Diffusion Posterior Sampling** framework.


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


## Usage

### Basic Usage

At first you need to sample the images and use DPS with changing the added noise which is a parameters in the inpainting_config.yaml you can run the sampling with running the run_with_mask which mask is not important. For the code running the jupyter notebook can be easy to read and follow. 
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


### Metrics

The following metrics are used for evaluation:
- **AUROC** (Area Under Receiver Operating Characteristic)




## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Base implementation from Diffusion Posterior Sampling
- [Add other acknowledgments]


---

**Last Updated**: March 2026
