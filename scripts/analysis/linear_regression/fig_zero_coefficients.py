from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from sou.experiments.linear_regression import dataset_dir, normalize_scenario, result_dir


# Figure 4 in the paper: Experiment 1, (n, p) = (405, 400),
# with s in {10, 50, 100, 200}. Only false positives above 1e-3 are shown.
EXPERIMENT = 1
SPEC = 4
REPS = [1, 5, 10, 20]
THRESHOLD = 1e-3
XLIM = (-1.0, 1.0)
BIN_WIDTH = 0.01
MODELS = [("lasso", "Lasso"), ("sou", "SoU-SGL")]


def load_zero_mask(sim_path: Path) -> np.ndarray:
    data = np.load(sim_path, allow_pickle=False)
    return ~np.asarray(data["beta_nonzero_mask"]).reshape(-1).astype(bool)


def load_all_coefs(run_files: List[Path]) -> np.ndarray:
    chunks = []
    for path in run_files:
        data = np.load(path, allow_pickle=True)
        if "coef" not in data:
            raise KeyError(
                f"{path} does not contain coefficients; rerun the model with --save-coef"
            )
        chunks.append(np.asarray(data["coef"], dtype=float))
    if not chunks:
        raise FileNotFoundError("No coefficient run files found")
    return np.vstack(chunks)


def symmetric_bins(xlim: Tuple[float, float], width: float) -> np.ndarray:
    max_abs = max(abs(xlim[0]), abs(xlim[1]))
    centers = np.arange(width / 2, max_abs + width, width)
    return np.concatenate([-centers[::-1], centers])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce the linear-regression false-positive figure from the paper."
    )
    p.add_argument("--results-root", default="results/linear_regression/runs")
    p.add_argument("--data-root", default="data/processed/regression")
    p.add_argument("--out", default="figures/fig_linear_regression_zero_coefficients.pdf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    data_root = Path(args.data_root)

    colors: Dict[str, str] = {"lasso": "#1b5e20", "sou": "#b22222"}
    values: Dict[Tuple[int, str], np.ndarray] = {}

    for rep in REPS:
        scenario = normalize_scenario(EXPERIMENT, SPEC, rep, None)
        sim0 = dataset_dir(data_root, scenario) / "sim_0000.npz"
        if not sim0.exists():
            raise FileNotFoundError(sim0)
        zero_mask = load_zero_mask(sim0)

        for model_key, _ in MODELS:
            run_dir = result_dir(results_root, model_key, scenario)
            run_files = sorted(run_dir.glob("**/runs_*.npz"))
            coefs = load_all_coefs(run_files)
            if coefs.shape[1] != zero_mask.size:
                raise RuntimeError(
                    f"Coefficient dimension mismatch for {model_key}, rep={rep}: "
                    f"{coefs.shape[1]} != {zero_mask.size}"
                )
            false_positives = coefs[:, zero_mask].reshape(-1)
            values[(rep, model_key)] = false_positives[
                np.abs(false_positives) > THRESHOLD
            ]

    bins = symmetric_bins(XLIM, BIN_WIDTH)
    y_max = 0.0
    for x in values.values():
        if x.size:
            hist, _ = np.histogram(x, bins=bins, density=True)
            y_max = max(y_max, float(np.max(hist)))
    y_max = max(y_max, 1.0)

    fig = plt.figure(figsize=(2.4 * len(REPS), 2.4))
    gs = gridspec.GridSpec(1, len(REPS), wspace=0.25)
    letters = "abcdefghijklmnopqrstuvwxyz"

    for j, rep in enumerate(REPS):
        ax = fig.add_subplot(gs[0, j])
        for model_key, label in MODELS:
            ax.hist(
                values[(rep, model_key)],
                bins=bins,
                density=True,
                color=colors[model_key],
                alpha=0.7,
                histtype="stepfilled",
                linewidth=0.0,
                label=label,
            )
        ax.set_xlim(*XLIM)
        ax.set_ylim(0.0, 1.05 * y_max)
        ax.set_yticks([])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_alpha(0.4)
        ax.spines["bottom"].set_alpha(0.4)
        ax.text(
            0.02,
            0.95,
            f"({letters[j]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_title(rf"$s={10 * rep}$", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.1),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
