import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from scipy.stats import norm


def set_up_plots():
      plt.rcParams.update({
        "font.family": "serif",       # Classic publication serif style
        "text.usetex": True,         # Set to True if your system has a full local LaTeX installation
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })
      
    
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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy.stats import norm

def plot_result_gmm(
    X,
    Y,
    logPOST,
    mus_hist,      # List or array containing the history of means over iterations
    Rs,
    mu0,           # Initial mean(s)
    weights=None,
    cmap="Blues",
    method="gvi"
):
    # ---------------------------------------------------------
    # Standardize inputs to handle both single Gaussian and GMM trajectories
    # ---------------------------------------------------------
    mus_hist = np.asarray(mus_hist)
    
    # If shape is (T, 2), it's a single component trajectory over T steps.
    # Convert it to (T, K, 2) where K = 1 for consistent looping.
    if mus_hist.ndim == 2:
        mus_hist = mus_hist[:, np.newaxis, :]
        
    num_components = mus_hist.shape[1]
    
    # Extract the final means from the last step of the history
    final_mus = mus_hist[-1] 
    
    # Standardize covariance factors (Rs)
    if num_components == 1 and np.ndim(Rs) == 2:
        Rs = [np.asarray(Rs)]
    else:
        Rs = [np.asarray(R) for R in Rs]
    
    # Standardize initial means (mu0)
    mu0 = np.asarray(mu0)
    if mu0.ndim == 1 and num_components == 1:
        mu0 = mu0[np.newaxis, :]

    # Handle component weights
    if weights is None:
        weights = np.ones(num_components) / num_components
    else:
        weights = np.asarray(weights, dtype=float)
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
    # Compute GMM Analytical Marginals (Using Final Means)
    # ---------------------------------------------------------
    px_gmm = np.zeros_like(x_vals)
    py_gmm = np.zeros_like(y_vals)

    for i in range(num_components):
        mu_final = final_mus[i]
        R = Rs[i]
        cov = R @ R.T
        w = weights[i]

        std_x = np.sqrt(cov[0, 0])
        std_y = np.sqrt(cov[1, 1])

        # Accumulate weighted 1D PDFs
        px_gmm += w * norm.pdf(x_vals, loc=mu_final[0], scale=std_x)
        py_gmm += w * norm.pdf(y_vals, loc=mu_final[1], scale=std_y)

    px_gmm /= np.max(px_gmm)
    py_gmm /= np.max(py_gmm)

    # ---------------------------------------------------------
    # Layout Setup
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(7, 6))
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
    # 2D Main Plot (Posterior, Trajectories, and Dashed Ellipses)
    # ---------------------------------------------------------
    ax_main.contourf(X, Y, POST, levels=20, cmap=cmap, alpha=0.9)
    ax_main.contour(X, Y, POST, levels=20, colors="k", linewidths=0.5, alpha=0.4)

    # 1. Plot Mean Trajectories for each Gaussian component
    colors_gmm = plt.cm.Reds(np.linspace(0.2, 1, max(num_components, 5)))
    for i in range(num_components):
        component_trajectory = mus_hist[:, i, :]
        ax_main.plot(
            component_trajectory[:, 0], 
            component_trajectory[:, 1], 
            color=colors_gmm[i], 
            linestyle="--", 
            alpha=0.5, 
            linewidth=1.5,
            zorder=3
        )

    # 2. Plot 1-sigma final ellipse with dashed red borders
    for i in range(num_components):
        mu_final = final_mus[i]
        R = Rs[i]
        cov = R @ R.T
        w = weights[i]

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(vals)
        fill_alpha = max(w * 1., 0.01)  # Keeps it clean and clear under lines

        ellipse = Ellipse(
            xy=mu_final,
            width=width,
            height=height,
            angle=theta,
            edgecolor=colors_gmm[i],                      # Red edge
            facecolor=(1.0, 0.0, 0.0, fill_alpha), 
            linewidth=1.5,
            linestyle="-",                       # Changed to dashed red
            label=method if i == 0 else "",
            zorder=4
        )
        ax_main.add_patch(ellipse)
        
        # Center of the components at termination step
        ax_main.scatter(mu_final[0], mu_final[1], color=colors_gmm[i], marker="x", s=35, alpha=1.0, zorder=5)
    
    # 3. Plot Initial Means with mu0 (Black Dots)
    for i in range(num_components):
        ax_main.scatter(mu0[i, 0], mu0[i, 1], color="black", marker="o", s=35, label="initial" if i == 0 else "", zorder=5)

    ax_main.set_xlabel(r"$x_1$")
    ax_main.set_ylabel(r"$x_2$")
    ax_main.set_xlim(x_vals.min(), x_vals.max())
    ax_main.set_ylim(y_vals.min(), y_vals.max())

    # ---------------------------------------------------------
    # Top Marginal Plot (x1)
    # ---------------------------------------------------------
    ax_top.plot(x_vals, px_true, lw=2, label="true")
    ax_top.plot(x_vals, px_gmm, lw=2, ls="--", label=method, color='red')

    ax_top.set_xlim(x_vals.min(), x_vals.max())
    ax_top.set_ylabel(r"$\pi(x_1)$")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # ---------------------------------------------------------
    # Right Marginal Plot (x2)
    # ---------------------------------------------------------
    ax_right.plot(py_true, y_vals, lw=2, label="true")
    ax_right.plot(py_gmm, y_vals, lw=2, ls="--", label=method, color='red')

    ax_right.set_ylim(y_vals.min(), y_vals.max())
    ax_right.set_xlabel(r"$\pi(x_2)$")
    ax_right.tick_params(axis="y", labelleft=False)

    plt.show()

def plot_optimization_trajectories(
    X,
    Y,
    logPOST,
    mus,          # Trajectory of means for 'natural' method (shape: [T, 2] or list)
    Rs,           # Final R matrix (or factor) for 'natural' method (shape: [2, 2])
    mus2,         # Trajectory of means for 'identity' method (shape: [T2, 2] or list)
    mus3,         # Trajectory of means for 'Newton-like' method (shape: [T3, 2] or list)
    mu0,          # Initial mean vector (shape: [2,])
    R0,           # Initial R matrix/covariance factor for the initial state (shape: [2, 2])
    cmap="Blues"
):
    """
    Plots the 2D true posterior contours with 1D marginal distributions along with
    the optimization trajectories of three methods: identity, natural, and Newton-like.
    Also displays the initial state uncertainty and final natural gradient uncertainty ellipses.
    """
    # Convert trajectory inputs to numpy arrays for reliable slicing
    mus = np.asarray(mus).squeeze()
    mus2 = np.asarray(mus2).squeeze()
    mus3 = np.asarray(mus3).squeeze()
    mu0 = np.asarray(mu0).squeeze()
    R0 = np.asarray(R0).squeeze()
    Rs = np.asarray(Rs).squeeze()

    # Extract final optimization state for the primary (natural) method
    final_mu_nat = mus[-1] if mus.ndim > 1 else mus
    final_mu_nat = final_mu_nat.flatten()
    # ---------------------------------------------------------
    # True Posterior (Grid Normalization)
    # ---------------------------------------------------------
    POST = np.exp(logPOST - np.max(logPOST))
    Z = np.sum(POST)
    if Z == 0 or not np.isfinite(Z):
        raise ValueError("Posterior normalization failed (underflow). Adjust your grid.")
    POST /= Z

    x_vals = X[0, :]
    y_vals = Y[:, 0]
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # True Marginals from grid evaluation
    py_true = np.sum(POST, axis=1) * dy
    px_true = np.sum(POST, axis=0) * dx

    px_true /= np.max(px_true)
    py_true /= np.max(py_true)

    # ---------------------------------------------------------
    # Compute Final Natural Method Analytical Marginals
    # ---------------------------------------------------------
    cov_nat = Rs @ Rs.T
    std_x_nat = np.sqrt(cov_nat[0, 0])
    std_y_nat = np.sqrt(cov_nat[1, 1])

    px_gmm = norm.pdf(x_vals, loc=final_mu_nat[0], scale=std_x_nat)
    py_gmm = norm.pdf(y_vals, loc=final_mu_nat[1], scale=std_y_nat)

    px_gmm /= np.max(px_gmm)
    py_gmm /= np.max(py_gmm)

    # ---------------------------------------------------------
    # Layout Setup
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(7., 6.))
    #fig = plt.figure(figsize=(9, 7))
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
    # 2D Main Plot (Posterior Contours & Trajectories)
    # ---------------------------------------------------------
    # Draw background true posterior contours
    ax_main.contourf(X, Y, POST, levels=16, cmap=cmap, alpha=0.9)
    ax_main.contour(X, Y, POST, levels=16, colors="k", linewidths=0.5, alpha=0.4)

    # 1. Plot Initial State (Green Marker and Green Ellipse)
    ax_main.scatter(mu0[0], mu0[1], color="black", marker="o", s=35, zorder=5)

    # 2. Plot Optimization Trajectories
    # Identity Trajectory (Orange)
    if mus2.ndim > 1:
        ax_main.plot(mus2[:, 0], mus2[:, 1], color="green", lw=1.7, linestyle="-", zorder=4)
        ax_main.scatter(mus2[-1, 0], mus2[-1, 1], color="green", marker="x", s=45, zorder=4)


    # Newton-like Trajectory (Purple)
    if mus3.ndim > 1:
        ax_main.plot(mus3[:, 0], mus3[:, 1], color="blue", lw=1.7, linestyle="-.", zorder=4)
        ax_main.scatter(mus3[-1, 0], mus3[-1, 1], color="blue", marker="x", s=45, zorder=4)

    
    # Natural Trajectory (Red)
    if mus.ndim > 1:
        ax_main.plot(mus[:, 0], mus[:, 1], color="red", lw=1.7, linestyle="--", zorder=4)
        ax_main.scatter(final_mu_nat[0], final_mu_nat[1], color="red", marker="x", s=45, zorder=4)

    # 3. Plot Final Ellipse ONLY for 'natural' (mus) trajectory
    vals, vecs = np.linalg.eigh(cov_nat)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)

    ellipse_final = Ellipse(
        xy=final_mu_nat, width=width, height=height, angle=theta,
        edgecolor=(1.0, 0.0, 0.0, 1.0),       # Solid Red matching its 1D marginal
        facecolor=(1.0, 0.0, 0.0, 0.5),      # Transparent Red interior
        linewidth=1.7, linestyle="-", zorder=3
    )
    ax_main.add_patch(ellipse_final)

    # Adjust Main Plot Aesthetics
    ax_main.set_xlabel(r"$x_1$")
    ax_main.set_ylabel(r"$x_2$")
    ax_main.set_xlim(x_vals.min(), x_vals.max())
    ax_main.set_ylim(y_vals.min(), y_vals.max())

    # ---------------------------------------------------------
    # Top Marginal Plot (x1 Profile)
    # ---------------------------------------------------------
    ax_top.plot(x_vals, px_true, lw=2, label="true")
    ax_top.plot(x_vals, px_gmm, lw=2, ls="--", label="Natural", color="red")
    ax_top.set_xlim(x_vals.min(), x_vals.max())
    ax_top.set_ylabel(r"$\pi(x_1)$")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    # ---------------------------------------------------------
    # Right Marginal Plot (x2 Profile)
    # ---------------------------------------------------------
    ax_right.plot(py_true, y_vals, lw=2, label="true")
    ax_right.plot(py_gmm, y_vals, lw=2, ls="--", label="natural", color="red")
    ax_right.set_ylim(y_vals.min(), y_vals.max())
    ax_right.set_xlabel(r"$\pi(x_2)$")
    ax_right.tick_params(axis="y", labelleft=False)

    
    