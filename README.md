# Sampling Toolbox

A small Python toolbox for particle-based sampling and Gaussian variational inference.

## Particle methods
* **Stein Variational Gradient Descent (SVGD)**
* **Unadjusted Langevin Algorithm (ULA)**
* **Affine-Invariant Langevin Interactive Dynamics (ALDI)**

## Variational inference
* **Mixture Gaussian variational inference**
  - Wasserstein gradient flow with different preconditioning strategies for means/covariance ODE system
  - Fisher-Rao gradient flow for the weights
  - adaptive time-stepping strategies
  - Cubature rule for expectations
  - Cholesky factor formulation for covariance matrices
  
## Installation

Clone the repository:

```bash
git clone <repository-url>
cd sampling_toolbox
```

Install the package:

```bash
pip install -e .
```

For development (including tests):

```bash
pip install -e ".[dev]"
```

## Examples
The `examples/` directory contains benchmark problems:

## References
- **SVGD:** Liu, Q. & Wang, D. (2016). *Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.* [arXiv](https://arxiv.org/abs/1608.04471)
- **ALDI:** A. Garbuno-Inigo, N. Nüsken, and S. Reich, Affine Invariant Interacting Langevin Dynamics for Bayesian Inference, SIAM Journal on Applied Dynamical Systems, 19(3), 1633–1658, 2020.
- **Gaussian WGF** M. Lambert, S. Chewi, F. Bach, S. Bonnabel, and P. Rigollet, Variational Inference via Wasserstein Gradient Flows, NeurIPS 35, 14434–14447, 2022.
- **Gradient-flow sampling** Y. Chen, D. Z. Huang, J. Huang, S. Reich, and A. M. Stuart, Sampling via Gradient Flows in the Space of Probability Measures, Mathematics of Computation, 2026.
