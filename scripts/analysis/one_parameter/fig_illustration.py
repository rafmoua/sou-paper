from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def posterior_normal_normal(mu0: float, sigma0: float, sigma: float, x: np.ndarray):
    """Posterior for mu with known sigma, prior N(mu0, sigma0^2), and likelihood N(mu, sigma^2)."""
    n = x.size
    xbar = float(np.mean(x))
    alpha = (1.0 / sigma0**2) / (1.0 / sigma0**2 + n / sigma**2)
    mu_n = alpha * mu0 + (1.0 - alpha) * xbar
    sigma_n = np.sqrt(1.0 / (1.0 / sigma0**2 + n / sigma**2))
    return mu_n, sigma_n


def posterior_sou_gamma(mu0: float, sigma0: float, sigma: float, gamma_mu: float, x: np.ndarray):
    """SoU posterior with gamma_mu confidence scaling."""
    n = x.size
    xbar = float(np.mean(x))
    alpha = (gamma_mu / sigma0**2) / (gamma_mu / sigma0**2 + n / sigma**2)
    mu_sou = alpha * mu0 + (1.0 - alpha) * xbar
    sigma_sou = np.sqrt(1.0 / (1.0 / sigma0**2 + n / (gamma_mu * sigma**2)))
    return mu_sou, sigma_sou


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the one-parameter illustration figure."""
    parser = argparse.ArgumentParser(description="Generate the one-parameter illustration figure.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for synthetic observations.")
    parser.add_argument("--out", type=str, default="figures/fig_one_parameter_illustration.pdf", help="Output figure path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mu0 = 0.0
    sigma = 1.0
    n = 12

    sigma0_values = [0.1, 0.2, 0.3, 0.6, 2.0]
    gamma_mu_values = [0.12, 0.4, 1.0, 4.0, 40.0]
    sigma0_fixed = 0.3

    out_pdf = Path(args.out)
    if not out_pdf.is_absolute():
        root = Path(__file__).resolve().parents[3]
        out_pdf = root / out_pdf
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(int(args.seed))
    x = np.random.normal(loc=3.0, scale=1.0, size=n)
    xbar = float(np.mean(x))

    mu_grid = np.linspace(-1, 4, 1000)
    like = norm.pdf(mu_grid, xbar, sigma / np.sqrt(n))

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), sharey=True)

    ax = axes[0]
    for idx, sigma0 in enumerate(sigma0_values):
        prior_color = plt.cm.Reds(1.0 - idx / len(sigma0_values))
        ax.plot(
            mu_grid,
            norm.pdf(mu_grid, mu0, sigma0),
            linestyle="--",
            color=prior_color,
            linewidth=1.8,
            alpha=0.75,
            label=f"Prior $\\sigma_0={sigma0}$",
        )

        mu_n, sig_n = posterior_normal_normal(mu0, sigma0, sigma, x)
        post_color = plt.cm.Blues(1.0 - idx / len(sigma0_values))
        ax.plot(
            mu_grid,
            norm.pdf(mu_grid, mu_n, sig_n),
            linestyle="-",
            color=post_color,
            linewidth=2.2,
            alpha=0.95,
            label=f"Post. $\\sigma_0={sigma0}$",
        )

    ax.plot(mu_grid, like, linestyle=":", color="black", linewidth=2.0, label="MLE")
    ax.axvline(xbar, color="black", linestyle="--", linewidth=1.4, label=r"$\delta_{\mathrm{MLE}}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Density")
    ax.set_xlim(-1, 4)
    ax.set_ylim(0, 5)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    ax = axes[1]
    ax.plot(
        mu_grid,
        norm.pdf(mu_grid, mu0, sigma0_fixed),
        linestyle="--",
        color=plt.cm.Reds(0.6),
        linewidth=1.8,
        alpha=0.75,
        label=f"Prior $\\sigma_0={sigma0_fixed}$",
    )

    for idx, gamma_mu in enumerate(gamma_mu_values):
        mu_sou, sig_sou = posterior_sou_gamma(mu0, sigma0_fixed, sigma, gamma_mu, x)
        post_color = plt.cm.Blues(1.0 - idx / len(gamma_mu_values))
        ax.plot(
            mu_grid,
            norm.pdf(mu_grid, mu_sou, sig_sou),
            linestyle="-",
            color=post_color,
            linewidth=2.2,
            alpha=0.95,
            label=rf"Post. $\gamma_\mu={gamma_mu}$",
        )

    ax.plot(mu_grid, like, linestyle=":", color="black", linewidth=2.0, label="MLE")
    ax.axvline(xbar, color="black", linestyle="--", linewidth=1.4, label=r"$\delta_{\mathrm{MLE}}$")
    ax.set_xlabel(r"$\mu$")
    ax.set_xlim(-1, 4)
    ax.set_ylim(0, 5)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    for axis in axes:
        axis.legend(loc="upper right", frameon=True, borderpad=0.4, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
