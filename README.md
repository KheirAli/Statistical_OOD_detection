# Statistical OOD Detection via Diffusion Posterior Sampling

This repository contains code for **statistical out-of-distribution (OOD) detection** using diffusion models, specifically leveraging **Diffusion Posterior Sampling (DPS)** for improved performance on inverse problems and uncertainty estimation in OOD scenarios.

## Overview

Out-of-distribution detection is critical for reliable deployment of machine learning models, especially in safety-critical applications. This project explores a statistical approach to OOD detection by utilizing diffusion models as powerful generative priors. 

We integrate **Diffusion Posterior Sampling** to better model the posterior distribution, enabling more robust uncertainty quantification and discrimination between in-distribution and out-of-distribution samples.

## Setup Instructions

### 1. Clone the Diffusion Posterior Sampling Repository

The core sampling logic is based on the official **Diffusion Posterior Sampling** implementation:

```bash
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git
