import numpy as np
from scipy.spatial.distance import pdist, squareform

def rbf_kernel(part, h=-1, kernel='rbf', p=2, scale=1.0):
    """
    Compute kernel matrix and its derivative.
    Parameters
    ----------
    scale : float
        Multiplicative factor applied to the median heuristic bandwidth.
        h = scale * sqrt(median(dist^2)/log(N))
    """
    sq_dist = pdist(part)
    pairwise_dists = squareform(sq_dist)**2

    if h < 0:
        med_sq = np.median(pairwise_dists)
        h = scale * np.sqrt(med_sq / np.log(part.shape[0]))

    if kernel == 'rbf':
        Kxy = np.exp(-pairwise_dists / (2.*h**2) )
        # compute derivative of Kernel: exp( -|x-x'|^2 / (2*h^2) ) wrt x
        # dxKxy = 1/h^2 * (x_j-x) * k(x, x_j)
        dxKxy = -np.matmul(Kxy, part)
        sumKxy = np.sum(Kxy, axis=1)
        for i in range(part.shape[1]): # sum over ndofs for all particles
            dxKxy[:, i] += np.multiply(part[:,i], sumKxy)
        dxKxy /= h**2
    
    elif kernel == 'p-power':
        # generalized RBF: exp(-||x - y||^p / (2h^p))
        Dp = (pairwise_dists**0.5)**p
        Kxy = np.exp(-Dp / (2. * h**p))
        # derivative: -p/(2h^p)*sum_j ((||x_i - x_j||^(p-2)) * (x_i - x_j) * K)
        diffs = part[:, None, :] - part[None, :, :]
        # avoid zero-distance division by adding identity
        norms = pairwise_dists + np.eye(part.shape[0])
        norms = norms**((p/2) - 1)
        term = p * norms / (2. * h**p)
        dx = np.einsum('ij,ij,ijk->ik', Kxy, term, diffs)
        dxKxy = dx

    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    return Kxy, dxKxy, h
