from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from sou.paths import bnn_mnist_runs_dir

matplotlib.use("Agg")

DEFAULT_LAYER1_KEY = "fc1"
DEFAULT_LAYER2_KEY = "fc_hidden.0"
DEFAULT_THR_IDX = 2  # idx 2 -> 1e-5

COLORS = {
    "bnn": "#1b5e20",
    "hs": "#1f4e79",
    "sou_sgl": "#b22222",
}

PRETTY = {
    "bnn": "BNN",
    "hs": "HS",
    "sou_sgl": "SoU-SGL",
}


def load_result_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_result_files(
    results_dir: Path,
    *,
    seed: int,
    num_layers: int,
    num_nodes: int,
    hshoe_scale: float,
    num_epochs: int,
    batch_size: int,
    step_size: float,
    variants: List[str],
) -> Dict[str, Path]:
    lr_tag = f"{float(step_size):.2e}"
    found: Dict[str, Path] = {}
    for variant in variants:
        filename = (
            f"mnist_bnn_variant={variant}"
            f"_layers={num_layers}"
            f"_nodes={num_nodes}"
            f"_hs={hshoe_scale}"
            f"_ep={num_epochs}"
            f"_bs={batch_size}"
            f"_lr={lr_tag}"
            f"_seed={seed}.json"
        )
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        found[variant] = path
    return found


def get_accuracy_series(result: Dict[str, Any]) -> pd.Series:
    history = result.get("history", {})
    accuracy = history.get("accuracy")
    if accuracy is None:
        raise ValueError("history['accuracy'] not found.")
    return pd.Series(accuracy, dtype=float)


def get_sparsity_series(
    result: Dict[str, Any],
    *,
    layer_key: str,
    thr_idx: int,
    source: str = "W_effective",
) -> pd.Series:
    history = result.get("history", {})
    sparsity = history.get("sparsity")
    if sparsity is None:
        raise ValueError("history['sparsity'] not found.")

    values: List[float] = []
    for epoch_dict in sparsity:
        if epoch_dict is None:
            values.append(float("nan"))
            continue
        if source not in epoch_dict or not isinstance(epoch_dict[source], dict):
            raise ValueError(f"Expected history['sparsity'][epoch]['{source}'] to be a dict.")
        layer_map = epoch_dict[source]
        if layer_key not in layer_map:
            raise KeyError(
                f"Layer key '{layer_key}' not found in sparsity dict. "
                f"Example keys: {list(layer_map.keys())[:10]}"
            )
        arr = layer_map[layer_key]
        if not isinstance(arr, (list, tuple)) or len(arr) <= thr_idx:
            raise ValueError(f"Unexpected sparsity format for key '{layer_key}': {arr}")
        values.append(float(arr[thr_idx]))
    return pd.Series(values, dtype=float)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def plot_final_figure(
    results: Dict[str, Dict[str, Any]],
    *,
    layer1_key: str,
    layer2_key: str,
    thr_idx: int,
    outpath: Path,
    rolling_window: int = 1,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        # figsize=(9.0, 4.8),
        figsize=(9.0, 3.8),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )

    ax = axes[0]
    for variant, result in results.items():
        series = get_accuracy_series(result)
        if rolling_window > 1:
            series = series.rolling(rolling_window, min_periods=1).mean()
        ax.plot(
            series.values,
            lw=1.6,
            color=COLORS.get(variant),
            alpha=0.95,
            label=PRETTY.get(variant, variant),
        )
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=2.2)
    panel_label(ax, "(a)")

    ax = axes[1]
    for variant, result in results.items():
        color = COLORS.get(variant)
        s1 = get_sparsity_series(result, layer_key=layer1_key, thr_idx=thr_idx, source="W_effective")
        s2 = get_sparsity_series(result, layer_key=layer2_key, thr_idx=thr_idx, source="W_effective")
        if rolling_window > 1:
            s1 = s1.rolling(rolling_window, min_periods=1).mean()
            s2 = s2.rolling(rolling_window, min_periods=1).mean()
        ax.plot(s1.values, lw=1.3, color=color, linestyle="-", alpha=0.9)
        ax.plot(s2.values, lw=1.3, color=color, linestyle="--", alpha=0.75)
    ax.set_ylabel("Sparsity")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    panel_label(ax, "(b)")

    handles: list[Line2D] = []
    labels: list[str] = []
    for variant in results.keys():
        handles.append(Line2D([0], [0], color=COLORS[variant], lw=1.6, linestyle="-"))
        labels.append(f"{PRETTY[variant]} | layer 1")
        handles.append(Line2D([0], [0], color=COLORS[variant], lw=1.6, linestyle="--"))
        labels.append(f"{PRETTY[variant]} | layer 2")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.24)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final MNIST BNN training dynamics figure (accuracy + combined layer sparsity)."
    )
    parser.add_argument("--results-dir", type=str, default=str(bnn_mnist_runs_dir()))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-nodes", type=int, default=200)
    parser.add_argument("--hshoe-scale", type=float, default=1e-1)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--layer1-key", type=str, default=DEFAULT_LAYER1_KEY)
    parser.add_argument("--layer2-key", type=str, default=DEFAULT_LAYER2_KEY)
    parser.add_argument("--thr-idx", type=int, default=DEFAULT_THR_IDX)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["bnn", "hs", "sou_sgl"],
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/fig_bnn_mnist_training_dynamics.pdf",
        help="Output figure path.",
    )
    parser.add_argument("--rolling", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_result_files(
        Path(args.results_dir),
        seed=int(args.seed),
        num_layers=int(args.num_layers),
        num_nodes=int(args.num_nodes),
        hshoe_scale=float(args.hshoe_scale),
        num_epochs=int(args.num_epochs),
        batch_size=int(args.batch_size),
        step_size=float(args.step_size),
        variants=list(args.variants),
    )
    results: Dict[str, Dict[str, Any]] = {
        variant: load_result_json(path) for variant, path in files.items()
    }

    plot_final_figure(
        results,
        layer1_key=str(args.layer1_key),
        layer2_key=str(args.layer2_key),
        thr_idx=int(args.thr_idx),
        outpath=Path(args.out),
        rolling_window=int(args.rolling),
    )


if __name__ == "__main__":
    main()
