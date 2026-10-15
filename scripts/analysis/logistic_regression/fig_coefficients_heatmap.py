from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sou.paths import logistic_regression_ovr_runs_dir

matplotlib.use("Agg")

MODELS = ["none", "l1cv", "gl", "sou"]
MODEL_NAMES = {
    "none": "LR",
    "l1cv": "LR-L1",
    "gl": "GL",
    "sou": "SoU-SGL",
}
DIGITS = list(range(10))
P = 784


def configure_plot_style() -> None:
    """Apply shared plotting style for logistic coefficient heatmaps."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_digit_npz(rawdir: Path, digit: int, seed: int) -> Dict[str, np.ndarray]:
    """Load one digit-specific OVR result file."""
    path = rawdir / f"mnist_ovr_digit={digit}_seed={seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_all_coefs(rawdir: Path, seed: int) -> np.ndarray:
    """Load coefficients for all models, digits and pixels into one array."""
    per_digit = {dig: load_digit_npz(rawdir, dig, seed) for dig in DIGITS}
    arr = np.zeros((len(MODELS), len(DIGITS), P), dtype=np.float64)

    for mi, m in enumerate(MODELS):
        key = f"coef_{m}"
        for di, dig in enumerate(DIGITS):
            coef = np.asarray(per_digit[dig][key], dtype=np.float64).reshape(-1)
            if coef.size != P:
                raise ValueError(f"Unexpected coef size digit={dig} model={m}: {coef.size}")
            arr[mi, di, :] = coef

    return arr


def robust_global_scale(coefs: np.ndarray, q: float = 0.995) -> float:
    """Compute a robust symmetric color scale from absolute coefficients."""
    vals = np.abs(coefs.reshape(-1))
    s = float(np.quantile(vals, q))
    if not np.isfinite(s) or s <= 0:
        mx = float(np.max(vals))
        s = mx if mx > 0 else 1.0
    return s


def nice_symmetric_ticks(v: float) -> List[float]:
    """Return symmetric colorbar ticks around zero."""
    return [-v, -0.5 * v, 0.0, 0.5 * v, v]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the logistic coefficient heatmap figure."""
    parser = argparse.ArgumentParser(description="Logistic OVR coefficient heatmaps with fixed global scale.")
    parser.add_argument("--rawdir", type=str, default=str(logistic_regression_ovr_runs_dir()), help="Directory with per-digit OVR NPZ files.")
    parser.add_argument("--seed", type=int, default=1, help="Seed identifier used in result file names.")
    parser.add_argument("--out", type=str, default="figures/fig_logistic_mnist_coefficients_heatmap.pdf", help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=600, help="Figure DPI.")
    parser.add_argument("--q", type=float, default=0.995, help="Quantile for robust fixed scale based on |coef|.")
    parser.add_argument("--figsize", type=str, default="16,6.8", help="Figure size as 'width,height'.")
    parser.add_argument("--cbar-label", type=str, default="Coefficient", help="Colorbar label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_plot_style()

    rawdir = Path(args.rawdir)
    out_path = Path(args.out)

    fs = [float(x.strip()) for x in args.figsize.split(",")]
    if len(fs) != 2:
        raise ValueError("--figsize must be like '16,6.8'")

    coefs = load_all_coefs(rawdir, seed=int(args.seed))
    s = robust_global_scale(coefs, q=float(args.q))

    fig, axes = plt.subplots(len(MODELS), len(DIGITS), figsize=(fs[0], fs[1]))

    im = None
    for r, m in enumerate(MODELS):
        for c, dig in enumerate(DIGITS):
            ax = axes[r, c]
            coef_img = coefs[r, c, :].reshape(28, 28)

            im = ax.imshow(
                coef_img,
                cmap="seismic",
                vmin=-s,
                vmax=s,
                interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(f"{dig}")

            if c == 0:
                ax.set_ylabel(
                    MODEL_NAMES[m],
                    rotation=0,
                    labelpad=35,
                    va="center",
                )

            ax.set_frame_on(False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(args.cbar_label)
    cb.set_ticks(nice_symmetric_ticks(s))
    cb.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.08, wspace=0.02, hspace=0.02)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=int(args.dpi))
    print(f"Saved: {out_path.resolve()}")

    plt.close(fig)


if __name__ == "__main__":
    main()
