# Physics-Enhanced Residual Learning (PERL)

This repository contains research code and experimental results associated with the Physics-Enhanced Residual Learning (PERL) framework for vehicle trajectory prediction.

PERL combines a physics-based car-following model with a learned residual correction. The framework is designed to preserve the interpretability of the physics model while improving prediction accuracy with data-driven learning.

## Associated Publication

K. Long, Z. Sheng, H. Shi, X. Li, S. Chen, and S. Ahn, “Physical enhanced residual learning (PERL) framework for vehicle trajectory prediction,” *Communications in Transportation Research*, vol. 5, 100166, 2025.  
https://doi.org/10.1016/j.commtr.2025.100166

## Core Implementation

The main PERL implementation is located in `models/PERL/`:

- `data.py`: data preparation and construction of physics-residual learning targets
- `train.py`: PERL model training
- `predict.py`: prediction and evaluation
- `model/`: trained model files and supporting artifacts

The `Physical_model/`, `Data_driven/`, and `PINN/` directories contain comparison methods used in the study. The `results/` and `results_figs/` directories contain experimental outputs and figures.

## Note

This repository preserves code developed across multiple stages of the research project. Some parameter settings and scripts reflect sensitivity analyses or later experimental variants and may not correspond exactly to a single configuration reported in the paper. It is provided for research transparency and reference.
