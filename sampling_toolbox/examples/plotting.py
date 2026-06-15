import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from scipy.stats import norm

def plot_kl(kl_tracks, method_name):
    plt.figure()
    plt.plot(kl_tracks[method_name], label=f'{method_name.upper()} KL Divergence')
    plt.xlabel('Iteration')
    plt.ylabel('KL')
    plt.legend()
    plt.show()

def plot_result(
    X,
    Y,
    logPOST,
    particles,
    vp_ref=None,
    cmap="Blues",
    method="svgd"
):

    # True posterior (grid)
    POST = np.exp(logPOST - np.max(logPOST))
    Z = np.sum(POST)
    if Z == 0 or not np.isfinite(Z):
        raise ValueError("Posterior normalization failed (underflow). Reduce grid range or scale log density.")
    POST /= Z

    x_vals = X[0, :]
    y_vals = Y[:, 0]
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # ---------------------------------------------------------
    # Marginals (true from grid)
    # ---------------------------------------------------------
    py_true = np.sum(POST, axis=1) * dy
    px_true = np.sum(POST, axis=0) * dx

    px_true /= np.max(px_true)
    py_true /= np.max(py_true)

    # marginals
    kde_x = gaussian_kde(particles[:, 0])
    x_grid = np.linspace(x_vals.min(), x_vals.max(), 300)
    px_svgd = kde_x(x_grid)
    px_svgd /= px_svgd.max()

    kde_y = gaussian_kde(particles[:, 1])
    y_grid = np.linspace(y_vals.min(), y_vals.max(), 300)
    py_svgd = kde_y(y_grid)
    py_svgd /= py_svgd.max()

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))

    gs = GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.15,
        wspace=0.15,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------
    # 2D posterior
    # ---------------------------------------------------------
    cf = ax_main.contourf(
        X, Y, POST,
        levels=20,
        cmap=cmap,
        alpha=0.9,
    )

    ax_main.contour(
        X, Y, POST,
        levels=20,
        colors="k",
        linewidths=0.5,
        alpha=0.4,
    )

    # SVGD particles
    ax_main.scatter(
        particles[:, 0],
        particles[:, 1],
        s=12,
        color="red",
        alpha=0.4,
        label=method,
    )

    # reference point (optional)
    if vp_ref is not None:
        ax_main.scatter(
            vp_ref[0],
            vp_ref[1],
            s=120,
            c="yellow",
            marker="*",
            edgecolors="k",
            zorder=10,
            label="reference",
        )

    ax_main.set_xlabel(r"$x_1$")
    ax_main.set_ylabel(r"$x_2$")
    ax_main.legend()

    ax_top.plot(x_vals, px_true, lw=2, label="true")
    ax_top.plot(x_grid, px_svgd, lw=2, ls="--", label=method, color='red')

    if vp_ref is not None:
        ax_top.axvline(vp_ref[0], color="black", ls="-.", alpha=0.6)

    ax_top.set_xlim(x_vals.min(), x_vals.max())
    ax_top.set_ylabel(r"$p(x_1)$")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend()

    # ---------------------------------------------------------
    # Right marginal (x2)
    # ---------------------------------------------------------
    ax_right.plot(py_true, y_vals, lw=2, label="true")
    ax_right.plot(py_svgd, y_grid, lw=2, ls="--", label=method, color='red')

    if vp_ref is not None:
        ax_right.axhline(vp_ref[1], color="black", ls="-.", alpha=0.6)

    ax_right.set_ylim(y_vals.min(), y_vals.max())
    ax_right.set_xlabel(r"$p(x_2)$")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.legend()

    #fig.colorbar(cf, ax=ax_main, fraction=0.046)
    plt.show()


def plot_result_gmm(
    X,
    Y,
    logPOST,
    mus,
    Rs,
    weights=None,  # Added weights parameter
    vp_ref=None,
    cmap="Blues",
    method="gvi"
):
    # ---------------------------------------------------------
    # Standardize inputs to handle both single Gaussian and GMM
    # ---------------------------------------------------------
    if np.ndim(mus) == 1:
        mus = [np.asarray(mus)]
        Rs = [np.asarray(Rs)]
        weights = [1.0]
    
    num_components = len(mus)
    
    # Handle weights if not provided or mismatching
    if weights is None:
        weights = np.ones(num_components) / num_components
    else:
        weights = np.asarray(weights, dtype=float)
        # Ensure weights sum to 1
        weights /= np.sum(weights)

    # True posterior (grid)
    POST = np.exp(logPOST - np.max(logPOST))
    Z = np.sum(POST)
    if Z == 0 or not np.isfinite(Z):
        raise ValueError("Posterior normalization failed (underflow). Reduce grid range or scale log density.")
    POST /= Z

    x_vals = X[0, :]
    y_vals = Y[:, 0]
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # Marginals (true from grid)
    py_true = np.sum(POST, axis=1) * dy
    px_true = np.sum(POST, axis=0) * dx

    px_true /= np.max(px_true)
    py_true /= np.max(py_true)

    # ---------------------------------------------------------
    # Compute GMM Analytical Marginals
    # ---------------------------------------------------------
    px_gmm = np.zeros_like(x_vals)
    py_gmm = np.zeros_like(y_vals)

    for i in range(num_components):
        mu = mus[i]
        R = Rs[i]
        cov = R @ R.T
        w = weights[i]

        # Extract marginal variances: var_x = cov[0,0], var_y = cov[1,1]
        std_x = np.sqrt(cov[0, 0])
        std_y = np.sqrt(cov[1, 1])

        # Accumulate weighted 1D PDFs
        px_gmm += w * norm.pdf(x_vals, loc=mu[0], scale=std_x)
        py_gmm += w * norm.pdf(y_vals, loc=mu[1], scale=std_y)

    # Normalize GMM marginal curves for visualization comparison
    px_gmm /= np.max(px_gmm)
    py_gmm /= np.max(py_gmm)

    # ---------------------------------------------------------
    # Layout Setup
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.15,
        wspace=0.15,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------
    # 2D Main Plot (Posterior Contours & Weighted Ellipses)
    # ---------------------------------------------------------
    cf = ax_main.contourf(X, Y, POST, levels=20, cmap=cmap, alpha=0.9)
    ax_main.contour(X, Y, POST, levels=20, colors="k", linewidths=0.5, alpha=0.4)

    # Plot 1-sigma ellipse for each Gaussian component
    for i in range(num_components):
        mu = mus[i]
        R = Rs[i]
        cov = R @ R.T
        w = weights[i]

        # Calculate eigenvalues and rotation angle for the ellipse
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are 2 * n_sigma * sqrt(eigenvalue)
        width, height = 2 * np.sqrt(vals)

        # Scale alpha directly with the weight component
        # Using max(..., 0.05) ensures even low-weight components remain slightly visible
        fill_alpha = max(w * 0.5, 0.05) 

        ellipse = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=theta,
            edgecolor=(1.0, 0.0, 0.0, 1.0),      # Solid Red (R, G, B, A)
            facecolor=(1.0, 0.0, 0.0, fill_alpha), # Transparent Red (R, G, B, A)
            linewidth=1.5,
            linestyle="-",
            label=method if i == 0 else "" 
        )
        ax_main.add_patch(ellipse)
        
        # Center of the components (marker alpha scales with weight too)
        ax_main.scatter(mu[0], mu[1], color="red", marker="x", s=20, alpha=1.0)

    # Reference point (optional)
    if vp_ref is not None:
        ax_main.scatter(
            vp_ref[0], vp_ref[1], s=120, c="yellow", marker="*",
            edgecolors="k", zorder=10, label="reference"
        )

    ax_main.set_xlabel(r"$x_1$")
    ax_main.set_ylabel(r"$x_2$")
    ax_main.legend()

    # ---------------------------------------------------------
    # Top Marginal Plot (x1)
    # ---------------------------------------------------------
    ax_top.plot(x_vals, px_true, lw=2, label="true")
    ax_top.plot(x_vals, px_gmm, lw=2, ls="--", label=method, color='red')

    if vp_ref is not None:
        ax_top.axvline(vp_ref[0], color="black", ls="-.", alpha=0.6)

    ax_top.set_xlim(x_vals.min(), x_vals.max())
    ax_top.set_ylabel(r"$p(x_1)$")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend()

    # ---------------------------------------------------------
    # Right Marginal Plot (x2)
    # ---------------------------------------------------------
    ax_right.plot(py_true, y_vals, lw=2, label="true")
    ax_right.plot(py_gmm, y_vals, lw=2, ls="--", label=method, color='red')

    if vp_ref is not None:
        ax_right.axhline(vp_ref[1], color="black", ls="-.", alpha=0.6)

    ax_right.set_ylim(y_vals.min(), y_vals.max())
    ax_right.set_xlabel(r"$p(x_2)$")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.legend()

    plt.show()

def plot_vi_diagnostics(means_hist, Rs_hist, weights_hist, kl_hist):
    """
    Plots the trajectories of means, covariance Frobenius norms, weights, and KL divergence.
    """
    steps = len(kl_hist)
    iterations = np.arange(steps)
    K = len(means_hist[0])
    dim = means_hist[0][0].shape[0]
    
    # Restructure histories for array slicing
    means_arr = np.array(means_hist)     # Shape: (steps, K, dim)
    weights_arr = np.array(weights_hist) # Shape: (steps, K)
    
    # Calculate Frobenius norm of Covariance matrices (Sigma = R @ R.T)
    cov_norms = np.zeros((steps, K))
    for s in range(steps):
        for k in range(K):
            R_mat = Rs_hist[s][k]
            Sigma = R_mat @ R_mat.T
            cov_norms[s, k] = np.linalg.norm(Sigma, ord='fro')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, max(K, 4)))

    # Subplot 1: Mean Trajectories
    for k in range(K):
        for d in range(dim):
            label = f"Comp {k} - Dim {d}" if d == 0 else ""
            axs[0, 0].plot(iterations, means_arr[:, k, d], color=colors[k], 
                           linestyle='-' if d == 0 else '--', label=label)
    axs[0, 0].set_title("Evolution of Component Means ($\mu$)")
    axs[0, 0].set_xlabel("Accepted Steps")
    axs[0, 0].set_ylabel("Position Value")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle=':')

    # Subplot 2: Covariance Frobenius Norms
    for k in range(K):
        axs[0, 1].plot(iterations, cov_norms[:, k], color=colors[k], lw=2, label=f"Comp {k}")
    axs[0, 1].set_title("Covariance Frobenius Norm ($\|\Sigma\|_F$)")
    axs[0, 1].set_xlabel("Accepted Steps")
    axs[0, 1].set_ylabel("Matrix Mass scale")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle=':')

    # Subplot 3: Weights Evolution
    for k in range(K):
        axs[1, 0].plot(iterations, weights_arr[:, k], color=colors[k], lw=2, label=f"Comp {k}")
    axs[1, 0].set_title("Component Weights Evolution ($w_k$)")
    axs[1, 0].set_xlabel("Accepted Steps")
    axs[1, 0].set_ylabel("Probability Mass")
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle=':')

    # Subplot 4: KL Divergence Metric
    axs[1, 1].plot(iterations, kl_hist, color='black', lw=2.5, label="$D_{KL}(q \parallel p)$")
    axs[1, 1].set_title("Kullback-Leibler Divergence Track")
    axs[1, 1].set_xlabel("Accepted Steps")
    axs[1, 1].set_ylabel("Nat Loss Error")
    #axs[1, 1].set_yscale('log')  # Log scale highlights late-stage fine tuning
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()