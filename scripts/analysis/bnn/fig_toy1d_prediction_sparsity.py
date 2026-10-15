from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from sou.paths import bnn_toy1d_runs_dir

matplotlib.use("Agg")

NUM_SAMPLES_LIST = [100, 150, 300]
NUM_NODES_LIST = [100, 200, 400]
MODEL_LIST = ["bnn", "hs", "sou_sgl"]
SEED = 1

TRUE_NOISE_STD = 0.2
XTEST_MIN, XTEST_MAX, XTEST_N = -4.0, 4.0, 200

DEFAULT_RESULTS_DIR = bnn_toy1d_runs_dir()
DEFAULT_OUTFILE = Path("figures") / "fig_bnn_toy1d_predictions_sparsity.pdf"

SUPPORTED_SPARSITY_THR = [1e-3, 1e-4, 1e-5]
SPARSITY_N_TRAIN = 300
SPARSITY_NODES = 200
DEFAULT_NUM_EPOCHS = 30000
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEP_SIZE = 0.005


def thr_to_tag(thr: float) -> str:
    """Map threshold value to the exact suffix used in run.npz keys."""
    if np.isclose(thr, 1e-3):
        return "0.001"
    if np.isclose(thr, 1e-4):
        return "0.0001"
    if np.isclose(thr, 1e-5):
        return "1e-05"
    raise ValueError(f"Unsupported thr={thr}. Supported: {SUPPORTED_SPARSITY_THR}")


def toy_dataset(n: int, noise_std: float, seed: int):
    """Generate the synthetic Toy1D dataset used in the experiments."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-4.0, 4.0, size=n)
    y_true = np.sin(x)
    y = y_true + rng.normal(0.0, noise_std, size=n)
    return x[:, None], y, y_true


def x_test_grid() -> np.ndarray:
    """Build the test grid used for predictive curves."""
    x = np.linspace(XTEST_MIN, XTEST_MAX, XTEST_N)[:, None]
    return x.astype(np.float32)


def _fmt_lr(step_size: float) -> str:
    return f"{float(step_size):.2e}"


def expected_run_path(
    results_dir: Path,
    model: str,
    n_train: int,
    nodes: int,
    seed: int,
    *,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    step_size: float = DEFAULT_STEP_SIZE,
) -> Path:
    """Return the expected run path for one Toy1D configuration."""
    return (
        results_dir
        / f"model={model}"
        / f"n={n_train}"
        / f"nodes={nodes}"
        / f"ep={num_epochs}_bs={batch_size}_lr={_fmt_lr(step_size)}"
        / f"seed={seed}"
        / "run.npz"
    )


def load_run_npz(path: Path) -> dict:
    """Load one Toy1D run file into a plain dictionary."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def extract_predictions_and_uncertainty(obj: dict):
    """Extract predictive mean and uncertainty components from one run object."""
    if "pred_mean" not in obj or "pred_std_epistemic" not in obj:
        raise KeyError("Expected keys {'pred_mean','pred_std_epistemic'} in run.npz")

    mean = np.asarray(obj["pred_mean"]).reshape(-1)
    std_epi = np.asarray(obj["pred_std_epistemic"]).reshape(-1)

    noise_var = obj.get("noise_var", np.nan)
    try:
        noise_var = float(np.asarray(noise_var).reshape(()))
    except Exception:
        noise_var = float("nan")

    std_ale = TRUE_NOISE_STD if np.isnan(noise_var) else float(np.sqrt(noise_var))
    return mean, std_epi, std_ale


def extract_sparsity_curve(run_obj: dict, thr: float = 1e-3, window: int = 50):
    """Extract smoothed sparsity curves for both layers at a fixed threshold."""
    tag = thr_to_tag(thr)
    key_layer1 = f"sparsity_layer1_thr_{tag}"
    key_layer2 = f"sparsity_layer2_thr_{tag}"

    if key_layer1 not in run_obj or key_layer2 not in run_obj:
        raise KeyError(f"Missing '{key_layer1}'/'{key_layer2}' in run.npz for thr={thr} (tag={tag}).")

    series_layer1 = pd.Series(np.asarray(run_obj[key_layer1], dtype=float))
    series_layer2 = pd.Series(np.asarray(run_obj[key_layer2], dtype=float))

    if window > 1:
        series_layer1 = series_layer1.rolling(window=window, min_periods=1).mean()
        series_layer2 = series_layer2.rolling(window=window, min_periods=1).mean()

    s1 = np.asarray(series_layer1, dtype=float)
    s2 = np.asarray(series_layer2, dtype=float)
    length = min(len(s1), len(s2))
    return np.arange(length), s1[:length], s2[:length]


def configure_style() -> None:
    """Apply shared plot style for Toy1D figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 13,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_prediction_panel(ax, run_obj: dict, n_train: int, x_test: np.ndarray) -> None:
    """Plot one prediction panel with train points and uncertainty bands."""
    x_train, y_train, _ = toy_dataset(n=n_train, noise_std=TRUE_NOISE_STD, seed=SEED)
    mean, std_epi, std_ale = extract_predictions_and_uncertainty(run_obj)

    x = x_test.squeeze()
    true_f = np.sin(x_test).squeeze()

    ax.scatter(
        x_train.squeeze(),
        y_train,
        s=6,
        color="black",
        alpha=0.65,
        label="Train",
        zorder=4,
        linewidths=0,
    )

    ax.plot(x, true_f, linestyle="--", color="red", lw=1.4, label="True function", zorder=5)
    ax.fill_between(
        x,
        true_f - 2 * TRUE_NOISE_STD,
        true_f + 2 * TRUE_NOISE_STD,
        color="red",
        alpha=0.18,
        label="True noise (+/-2sigma)",
        zorder=1,
    )

    ax.plot(x, mean, color="blue", lw=1.6, label="Predictive mean", zorder=6)
    ax.fill_between(
        x,
        mean - 2 * std_ale,
        mean + 2 * std_ale,
        color="blue",
        alpha=0.28,
        label="Aleatoric (+/-2sigma)",
        zorder=2,
    )
    ax.fill_between(
        x,
        mean - 2 * (std_ale + std_epi),
        mean + 2 * (std_ale + std_epi),
        color="blue",
        alpha=0.14,
        label="Total (+/-2sigma)",
        zorder=0,
    )

    ax.set_ylim(-2.2, 2.2)
    ax.set_xlim(x.min(), x.max())
    ax.set_xticks([-3, 0, 3])
    ax.set_xticklabels(["-3", "0", "3"])


def make_figure(
    results_dir: Path,
    out_pdf: Path,
    sparsity_thr: float = 1e-3,
    window: int = 50,
    *,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    step_size: float = DEFAULT_STEP_SIZE,
) -> None:
    """Create the full Toy1D figure with prediction grid and sparsity row."""
    configure_style()
    x_test = x_test_grid()

    n_samples = len(NUM_SAMPLES_LIST)
    n_nodes = len(NUM_NODES_LIST)
    n_models = len(MODEL_LIST)
    total_rows = n_samples + 1

    block_space = 0.10
    width_ratios = []
    for model_index in range(n_models):
        width_ratios += [1.0] * n_nodes
        if model_index < n_models - 1:
            width_ratios.append(block_space)

    height_ratios = [1.0] * total_rows

    fig_w = 2.32 * (n_nodes * n_models + (n_models - 1) * 0.25)
    fig_h = 12.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        total_rows,
        len(width_ratios),
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.03,
        hspace=0.18,
    )

    model_label = {"bnn": "BNN", "hs": "HS", "sou_sgl": "SoU-SGL"}

    prediction_legend_axis = None
    top_axes = {}
    sparsity_axes = {}

    for sample_row, n_train in enumerate(NUM_SAMPLES_LIST):
        for model_block, model in enumerate(MODEL_LIST):
            start_col = model_block * (n_nodes + 1)

            for node_col, nodes in enumerate(NUM_NODES_LIST):
                col = start_col + node_col
                run_path = expected_run_path(
                    results_dir,
                    model=model,
                    n_train=n_train,
                    nodes=nodes,
                    seed=SEED,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    step_size=step_size,
                )
                if not run_path.exists():
                    raise FileNotFoundError(f"Missing: {run_path}")
                run_obj = load_run_npz(run_path)

                ax = fig.add_subplot(gs[sample_row, col])
                plot_prediction_panel(ax, run_obj, n_train=n_train, x_test=x_test)

                if col == 0:
                    ax.set_ylabel(f"$n={n_train}$", fontsize=14, labelpad=10)
                else:
                    ax.set_yticks([])

                if sample_row < n_samples - 1:
                    ax.set_xticklabels([])

                if prediction_legend_axis is None:
                    prediction_legend_axis = ax

                if sample_row == 0:
                    top_axes[(model, nodes)] = ax

    sparsity_color_layer1 = "#3A3A3A"
    sparsity_color_layer2 = "#9A9A9A"
    ls_layer1 = "-"
    ls_layer2 = "--"

    for model_block, model in enumerate(MODEL_LIST):
        start_col = model_block * (n_nodes + 1)
        end_col = start_col + n_nodes - 1
        ax = fig.add_subplot(gs[n_samples, start_col : end_col + 1])

        run_path = expected_run_path(
            results_dir,
            model=model,
            n_train=SPARSITY_N_TRAIN,
            nodes=SPARSITY_NODES,
            seed=SEED,
            num_epochs=num_epochs,
            batch_size=batch_size,
            step_size=step_size,
        )
        if not run_path.exists():
            raise FileNotFoundError(f"Missing: {run_path}")
        run_obj = load_run_npz(run_path)

        epochs, sparsity_layer1, sparsity_layer2 = extract_sparsity_curve(run_obj, thr=sparsity_thr, window=window)

        ax.plot(epochs, sparsity_layer1, lw=1.2, color=sparsity_color_layer1, linestyle=ls_layer1, label="Layer 1")
        ax.plot(epochs, sparsity_layer2, lw=1.2, color=sparsity_color_layer2, linestyle=ls_layer2, label="Layer 2")

        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, max(int(epochs.max()), 1))
        ax.set_xlabel("Epoch")

        if model_block == 0:
            ax.set_ylabel("Sparsity", fontsize=14, labelpad=10)
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

        sparsity_axes[model] = ax

    left, right = 0.05, 0.98
    top, bottom = 0.88, 0.17
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    fig.canvas.draw()

    for model_block, model in enumerate(MODEL_LIST):
        for nodes in NUM_NODES_LIST:
            ax = top_axes[(model, nodes)]
            bb = ax.get_position()
            x_mid = 0.5 * (bb.x0 + bb.x1)
            fig.text(x_mid, top + 0.008, f"{nodes} nodes", ha="center", va="bottom", fontsize=13)

    for model in MODEL_LIST:
        bbs = [top_axes[(model, nodes)].get_position() for nodes in NUM_NODES_LIST]
        x0 = min(bb.x0 for bb in bbs)
        x1 = max(bb.x1 for bb in bbs)
        x_mid = 0.5 * (x0 + x1)
        fig.text(x_mid, top + 0.038, model_label[model], ha="center", va="bottom", fontsize=16, fontweight="bold")

    bb0 = top_axes[(MODEL_LIST[0], NUM_NODES_LIST[0])].get_position()
    fig.text(bb0.x0 - 0.020, bb0.y1 + 0.010, "(a)", fontsize=14, fontweight="bold")

    bb_s = sparsity_axes[MODEL_LIST[0]].get_position()
    fig.text(bb_s.x0 - 0.020, bb_s.y1 + 0.010, "(b)", fontsize=14, fontweight="bold")

    pred_handles, pred_labels = prediction_legend_axis.get_legend_handles_labels()

    sparsity_handles = [
        Line2D([0], [0], color=sparsity_color_layer1, lw=1.6, linestyle=ls_layer1, label="Layer 1"),
        Line2D([0], [0], color=sparsity_color_layer2, lw=1.6, linestyle=ls_layer2, label="Layer 2"),
    ]
    sparsity_labels = [h.get_label() for h in sparsity_handles]

    fig.legend(
        pred_handles,
        pred_labels,
        loc="lower center",
        bbox_to_anchor=(0.25, 0.067),
        ncol=3,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.25,
    )
    fig.legend(
        sparsity_handles,
        sparsity_labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.067),
        ncol=2,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.35,
        # title=rf"Sparsity ($n={SPARSITY_N_TRAIN}$, nodes={SPARSITY_NODES}$)",
        title=rf"Sparsity ($n={SPARSITY_N_TRAIN},\ \mathrm{{nodes}}={SPARSITY_NODES}$)",
        title_fontsize=14,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved: {out_pdf}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Toy1D prediction+sparsity figure."""
    parser = argparse.ArgumentParser(
        description="Plot Toy1D predictions and sparsity trajectories for BNN variants."
    )
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Root directory with Toy1D run outputs.")
    parser.add_argument("--out-pdf", type=str, default=str(DEFAULT_OUTFILE), help="Output figure path.")
    parser.add_argument("--thr", type=float, default=1e-5, help="Sparsity threshold to visualize.")
    parser.add_argument("--window", type=int, default=1, help="Rolling-average window for sparsity curves.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Training epochs used in run path.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size used in run path.")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE, help="Step size used in run path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(
        results_dir=Path(args.results_dir),
        out_pdf=Path(args.out_pdf),
        sparsity_thr=float(args.thr),
        window=int(args.window),
        num_epochs=int(args.num_epochs),
        batch_size=int(args.batch_size),
        step_size=float(args.step_size),
    )


if __name__ == "__main__":
    main()

# py scripts\analysis\bnn\fig_toy1d_prediction_sparsity.py --out-pdf figures\fig_bnn_toy1d_predictions_sparsity.pdf
