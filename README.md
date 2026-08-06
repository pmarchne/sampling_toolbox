# Sampling Toolbox

A Python toolbox for particle-based sampling and variational inference methods.

## Implemented methods

### Particle methods

* **Stein Variational Gradient Descent (SVGD)**
* **Unadjusted Langevin Algorithm (ULA)**
* **Affine-Invariant Langevin Interactive Dynamics (ALDI)**

### Variational inference methods

* **Mixture Gaussian variational inference based on gradient flow**

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

## Quick example

```python
import numpy as np
from sampling_toolbox import SVGD

# Define log posterior and gradients
def log_likelihood(x):
    return -0.5 * np.dot(x, x)

def log_prior(x):
    return 0.0

def grad_log_likelihood(x):
    return -x

def grad_log_prior(x):
    return np.zeros_like(x)

sampler = SVGD(
    log_likelihood,
    log_prior,
    grad_log_likelihood,
    grad_log_prior,
    step_size=0.1,
    n_iter=100
)

particles = np.random.randn(100, 2)

samples, history, kl = sampler._sample(particles)
```

## Examples

The `examples/` directory contains benchmark problems:

* Gaussian targets
* Rosenbrock distributions
* comparisons between sampling methods
* variational inference demonstrations


## License

Private research project.
