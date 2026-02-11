# Structured Uncertainty Prediction for Brain MRI Segmentation

## Overview

This repository contains the implementation for my undergraduate thesis project on **uncertainty modelling in brain tumour MRI segmentation**.

The project investigates whether a Structured Uncertainty Prediction Network (SUPN) can approximate the predictive distribution of a segmentation ensemble by learning a Gaussian distribution in logit space, parameterised by a mean and a structured precision (inverse covariance) matrix.

The goal was to explore whether such a model could provide:

- A formal representation of predictive uncertainty  
- Spatially correlated uncertainty estimates  
- Efficient sampling of plausible segmentation masks  

The approach was evaluated on 2D slices from the BraTS 2020 dataset.

This repository reflects an exploratory research project. While the framework was successfully implemented, empirical results were limited and highlight important challenges in applying structured uncertainty modelling to medical segmentation.

---

## Motivation

Deep segmentation models typically produce deterministic outputs. In medical imaging, however, segmentation boundaries are often ambiguous due to:

- Inter-rater variability  
- Imaging noise  
- Tumour heterogeneity  

Common uncertainty estimation approaches include:

- Deep ensembles  
- Monte Carlo dropout  
- Latent-variable models (e.g. Probabilistic U-Net)

This project explores an alternative approach: directly learning a structured Gaussian distribution over segmentation logits, following prior work on SUPNs.

---

## Method Summary

Two model families were implemented:

### 1. Ensemble U-Net (Baseline)

Multiple U-Net models were trained independently to form an ensemble.  
The ensemble predictions were used to estimate an empirical predictive distribution in logit space.

### 2. Structured Uncertainty Prediction Network (SUPN)

The SUPN model:

- Predicts mean logits
- Predicts a sparse precision matrix via Cholesky parameterisation
- Models a multivariate Gaussian in logit space
- Enables sampling of spatially correlated segmentation masks

Training was performed in two stages:
1. Matching ensemble mean logits (MSE)
2. Fitting the predictive distribution via negative log-likelihood

---

## Results Summary

The ensemble baseline achieved moderate segmentation performance but exhibited variability across tumour types and slice locations.

The SUPN successfully learned a structured distribution and enabled sampling; however:

- Sampled segmentations often contained artifacts  
- Performance was strongly dependent on ensemble quality  
- Full covariance modelling proved computationally challenging  
- Generalisation remained limited  

These findings suggest that while structured uncertainty modelling is conceptually appealing, its practical application in this setting requires further methodological refinement and stronger ensemble baselines.

---

## Limitations

- Downsampled 2D inputs (64×64 resolution)
- Binary segmentation (tumour vs background)
- Strong dependence on ensemble performance
- Limited hyperparameter search
- Computational constraints on covariance modelling

This implementation should be viewed as a research prototype rather than a production-ready medical system.

---

## Installation

### Docker

```bash
docker run supn_cholespy_image
```

### Option 2: Manual Installation

Install required dependencies:

``` bash
pip install wandb matplotlib opencv-python scikit-image pandas
```

Additional PyTorch and CUDA dependencies may be required depending on
your setup.

------------------------------------------------------------------------

## Training

To train the model:

``` bash
python train.py
```

Device configuration (CPU/GPU) is handled in:

    devices.py

------------------------------------------------------------------------

## Project Structure

    .
    ├── train.py                     # Training pipeline
    ├── u_net.py                     # U-Net backbone
    ├── sampling.py                  # Structured Gaussian sampling utilities
    ├── metrics.py                   # Core metrics (IoU, accuracy)
    ├── ensemble_metrics.py          # Ensemble comparison metrics
    ├── compute_stats.py             # Dataset statistics
    ├── devices.py                   # Device configuration
    ├── synth_data.py                # Synthetic data experiments
    │
    ├── models/
    │   ├── model.py                 # Primary SUPN model
    │   ├── model2.py                # Alternate model variant
    │   └── supn_blocks.py           # Structured covariance blocks
    │
    ├── notebooks/
    │   ├── data_loader.py
    │   ├── data_sanity_check.py
    │   ├── data_split.py
    │   ├── model_pred.py
    │   └── tester.py

------------------------------------------------------------------------
