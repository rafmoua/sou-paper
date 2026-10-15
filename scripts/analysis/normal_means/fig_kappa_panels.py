from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sou.paths import normal_means_runs_dir

matplotlib.use("Agg")


def ideal_sparse_reference(n_total_samples: int, n_nonzero: int, n_zero: int) -> np.ndarray:
    """Build an ideal two-point kappa reference with matching class proportions."""
    n_nz = int(round(n_total_samples * n_nonzero / (n_nonzero + n_zero)))
    n_z = int(round(n_total_samples * n_zero / (n_nonzero + n_zero)))
    if n_nz + n_z != n_total_samples:
        n_z = n_total_samples - n_nz
    return np.concatenate([np.zeros(n_nz), np.ones(n_z)])


def configure_plot_style() -> None:
    """Apply shared style for normal-means figures."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_labels(axes, labels=("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")) -> None:
    """Add panel labels to a list of axes."""
    for ax, label in zip(axes, labels):
        ax.text(
            0.03,
            0.97,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
        )


def style_axes(ax: plt.Axes) -> None:
    """Apply per-axis style formatting."""
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\kappa$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_hist_bars(
    ax,
    data: np.ndarray,
    *,
    bins: np.ndarray,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    label: str | None = None,
    hatch: str | None = None,
    zorder: int = 2,
):
    """Plot one histogram panel using bar style and optional hatching."""
    _, _, patches = ax.hist(
        np.asarray(data),
        bins=bins,
        density=True,
        histtype="bar",
        color=facecolor,
        edgecolor=edgecolor,
        linewidth=0.6,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )
    if hatch is not None:
        for patch in patches:
            patch.set_hatch(hatch)


def plot_kappa_panels_bars(
    *,
    kappa_sou_nonzero: np.ndarray,
    kappa_sou_zero: np.ndarray,
    kappa_hs_nonzero: np.ndarray,
    kappa_hs_zero: np.ndarray,
    n_nonzero: int,
    n_zero: int,
    bins: int = 50,
    show_ideal: bool = True,
    free_y: bool = True,
):
    """Create the full 2x3 kappa panel figure."""
    configure_plot_style()

    kappa_sou_all = np.concatenate([kappa_sou_nonzero, kappa_sou_zero])
    kappa_hs_all = np.concatenate([kappa_hs_nonzero, kappa_hs_zero])
    ideal_data = ideal_sparse_reference(len(kappa_sou_all), n_nonzero, n_zero)

    color_sou = "#b22222"
    color_hs = "#1f4e79"
    color_ideal = "0.70"

    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    fig = plt.figure(figsize=(14.0, 6.4))
    gs = gridspec.GridSpec(2, 3, hspace=0.32, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    axes_all = [ax1, ax2, ax3, ax4, ax5, ax6]

    ax1.set_title("Strong", fontweight="bold")
    ax2.set_title("Null", fontweight="bold")
    ax3.set_title("Aggregate", fontweight="bold")

    ax1.text(-0.18, 0.5, "SoU-SGL", transform=ax1.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
    ax4.text(-0.18, 0.5, "Horseshoe", transform=ax4.transAxes, rotation=90, va="center", ha="center", fontweight="bold")

    ax1.set_ylabel("Density")
    ax4.set_ylabel("Density")

    plot_hist_bars(ax1, kappa_sou_nonzero, bins=bin_edges, facecolor=color_sou, edgecolor=color_sou, alpha=0.45)
    plot_hist_bars(ax2, kappa_sou_zero, bins=bin_edges, facecolor=color_sou, edgecolor=color_sou, alpha=0.45)

    if show_ideal:
        plot_hist_bars(
            ax3,
            ideal_data,
            bins=bin_edges,
            facecolor=color_ideal,
            edgecolor="0.55",
            alpha=0.35,
            hatch="///",
            label="Ideal",
        )
    plot_hist_bars(ax3, kappa_sou_all, bins=bin_edges, facecolor=color_sou, edgecolor=color_sou, alpha=0.45, label="SoU")

    plot_hist_bars(ax4, kappa_hs_nonzero, bins=bin_edges, facecolor=color_hs, edgecolor=color_hs, alpha=0.45)
    plot_hist_bars(ax5, kappa_hs_zero, bins=bin_edges, facecolor=color_hs, edgecolor=color_hs, alpha=0.45)

    if show_ideal:
        plot_hist_bars(
            ax6,
            ideal_data,
            bins=bin_edges,
            facecolor=color_ideal,
            edgecolor="0.55",
            alpha=0.35,
            hatch="///",
            label="Ideal",
        )
    plot_hist_bars(ax6, kappa_hs_all, bins=bin_edges, facecolor=color_hs, edgecolor=color_hs, alpha=0.45, label="Horseshoe")

    for ax in axes_all:
        style_axes(ax)

    if not free_y:
        pass

    add_panel_labels(axes_all)

    ax3.legend(loc="upper left", bbox_to_anchor=(0.12, 0.98), frameon=True, borderpad=0.35, handlelength=1.6)
    ax6.legend(loc="upper left", bbox_to_anchor=(0.12, 0.98), frameon=True, borderpad=0.35, handlelength=1.6)

    return fig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the normal-means kappa figure."""
    parser = argparse.ArgumentParser(description="Plot normal-means kappa panels from saved samples.")
    parser.add_argument("--npz", type=str, default=str(normal_means_runs_dir() / "kappa_samples.npz"), help="Input NPZ with kappa samples.")
    parser.add_argument("--out", type=str, default="figures/fig_normal_means_kappa_panels.pdf", help="Output figure path.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins on [0, 1].")
    parser.add_argument("--no_ideal", action="store_true", help="Disable ideal reference distribution overlay.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = np.load(args.npz, allow_pickle=False)
    n_nonzero = int(data["n_nonzero"])
    if "n_zero" in data.files:
        n_zero = int(data["n_zero"])
    elif "n_total" in data.files:
        n_zero = int(data["n_total"]) - n_nonzero
    else:
        # Last-resort inference from the number of posterior kappa draws.
        n_zero = int(round(len(data["kappa_sou_zero"]) / (len(data["kappa_sou_nonzero"]) / n_nonzero)))

    fig = plot_kappa_panels_bars(
        kappa_sou_nonzero=data["kappa_sou_nonzero"],
        kappa_sou_zero=data["kappa_sou_zero"],
        kappa_hs_nonzero=data["kappa_hs_nonzero"],
        kappa_hs_zero=data["kappa_hs_zero"],
        n_nonzero=n_nonzero,
        n_zero=n_zero,
        bins=args.bins,
        show_ideal=not args.no_ideal,
        free_y=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
