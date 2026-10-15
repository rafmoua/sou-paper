from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sou.paths import bnn_tables_dir, bnn_toy1d_runs_dir

MODEL_LIST = ["bnn", "hs", "sou_sgl"]
NODES_LIST = [100, 200, 400]
TRUE_NOISE_STD = 0.2
SUPPORTED_SPARSITY_THR = [1e-3, 1e-4, 1e-5]
DEFAULT_NUM_EPOCHS = 30000
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEP_SIZE = 0.005


METRIC_SPECS = [
    ("std_aleatoric", "Aleatoric uncertainty", "min"),
    ("std_epistemic", "Epistemic uncertainty", "min"),
    ("sparsity_layer1", "Layer 1 weight sparsity", "max"),
    ("sparsity_layer2", "Layer 2 weight sparsity", "max"),
]


def thr_to_tag(thr: float) -> str:
    """Map threshold value to the exact suffix used in saved run keys."""
    if np.isclose(thr, 1e-3):
        return "0.001"
    if np.isclose(thr, 1e-4):
        return "0.0001"
    if np.isclose(thr, 1e-5):
        return "1e-05"
    raise ValueError(f"Unsupported thr={thr}. Supported: {SUPPORTED_SPARSITY_THR}")


def thr_to_tex(thr: float) -> str:
    """Format threshold for LaTeX captions."""
    exponent = int(np.round(np.log10(thr)))
    if np.isclose(thr, 10.0 ** exponent):
        return rf"10^{{{exponent}}}"
    return rf"{thr:.0e}"


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
    """Return the expected run path for one model/nodes configuration."""
    return (
        results_dir
        / f"model={model}"
        / f"n={n_train}"
        / f"nodes={nodes}"
        / f"ep={num_epochs}_bs={batch_size}_lr={_fmt_lr(step_size)}"
        / f"seed={seed}"
        / "run.npz"
    )


def safe_scalar(x, default=np.nan) -> float:
    """Convert an array-like scalar to float with a fallback default."""
    try:
        return float(np.asarray(x).reshape(()))
    except Exception:
        return float(default)


def load_npz(path: Path) -> dict:
    """Load one NPZ file into a plain dictionary."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_sparsity_finals(run_obj: dict, thr: float) -> Tuple[float, float]:
    """Extract final layer-1/layer-2 sparsity at a selected threshold."""
    tag = thr_to_tag(thr)

    k1_final = f"sparsity_layer1_final_thr_{tag}"
    k2_final = f"sparsity_layer2_final_thr_{tag}"
    k1_hist = f"sparsity_layer1_thr_{tag}"
    k2_hist = f"sparsity_layer2_thr_{tag}"

    if k1_final in run_obj and k2_final in run_obj:
        return safe_scalar(run_obj[k1_final], default=np.nan), safe_scalar(run_obj[k2_final], default=np.nan)

    if k1_hist in run_obj and k2_hist in run_obj:
        s1_arr = np.asarray(run_obj[k1_hist], dtype=float).reshape(-1)
        s2_arr = np.asarray(run_obj[k2_hist], dtype=float).reshape(-1)
        s1 = float(s1_arr[~np.isnan(s1_arr)][-1]) if np.any(~np.isnan(s1_arr)) else np.nan
        s2 = float(s2_arr[~np.isnan(s2_arr)][-1]) if np.any(~np.isnan(s2_arr)) else np.nan
        return s1, s2

    return np.nan, np.nan


def get_std_aleatoric(run_obj: dict) -> float:
    """Extract aleatoric uncertainty scalar from a run object."""
    noise_var = safe_scalar(run_obj.get("noise_var", np.nan), default=np.nan)
    if np.isnan(noise_var):
        return float(TRUE_NOISE_STD)
    return float(np.sqrt(noise_var))


def get_std_epistemic(run_obj: dict) -> float:
    """Extract epistemic uncertainty scalar from a run object."""
    if "std_epistemic_scalar" in run_obj:
        return safe_scalar(run_obj["std_epistemic_scalar"], default=np.nan)
    if "pred_std_epistemic" in run_obj:
        return float(np.asarray(run_obj["pred_std_epistemic"], dtype=float).mean())
    return np.nan


def pretty_model_name(model: str) -> str:
    """Convert internal model key to paper display name."""
    return {"bnn": "BNN", "hs": "HS", "sou_sgl": "SoU"}[model]


def _best_mask(values: List[float], mode: str, tol: float = 1e-12) -> List[bool]:
    arr = np.asarray(values, dtype=float)
    if np.all(~np.isfinite(arr)):
        return [False] * len(values)
    if mode == "min":
        best = np.nanmin(arr)
        return [np.isfinite(v) and abs(v - best) <= tol for v in values]
    if mode == "max":
        best = np.nanmax(arr)
        return [np.isfinite(v) and abs(v - best) <= tol for v in values]
    raise ValueError(mode)


def _fmt(x: float, ndigits: int) -> str:
    return f"{x:.{ndigits}f}"


def to_latex_table_toy1d(
    wide: pd.DataFrame,
    *,
    models_tex: List[str],
    nodes_list: List[int],
    caption: str,
    label: str,
    digits: int = 3,
) -> str:
    """Render the Toy1D summary table in LaTeX format."""
    if not isinstance(wide.columns, pd.MultiIndex):
        raise ValueError("Expected wide columns to be a MultiIndex with levels [Nodes, Model].")

    colspec = "@{} l " + " ".join(["S[table-format=1.3]"] * (len(nodes_list) * len(models_tex))) + " @{}"

    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\footnotesize")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    groups = " & ".join([rf"\multicolumn{{{len(models_tex)}}}{{c}}{{{nodes} Nodes}}" for nodes in nodes_list])
    lines.append(rf"\multirow{{2}}{{*}}{{Metric}} & {groups} \\")

    start = 2
    cmid = []
    for group_idx in range(len(nodes_list)):
        left = start + group_idx * len(models_tex)
        right = left + len(models_tex) - 1
        cmid.append(rf"\cmidrule(lr){{{left}-{right}}}")
    lines.append(" ".join(cmid))

    subheader = " & ".join(models_tex * len(nodes_list))
    lines.append(rf" & {subheader} \\")
    lines.append(r"\midrule")

    for metric_key, row_label, better in METRIC_SPECS:
        parts = [row_label]

        for nodes in nodes_list:
            values = [float(wide.loc[metric_key, (nodes, model)]) for model in models_tex]
            best = _best_mask(values, mode=better)
            for value, is_best in zip(values, best):
                if np.isnan(value):
                    parts.append(r"{\text{--}}")
                else:
                    txt = _fmt(value, digits)
                    parts.append(rf"\textbf{{{txt}}}" if is_best else txt)

        lines.append(" & ".join(parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Toy1D summary table."""
    parser = argparse.ArgumentParser(description="Generate Toy1D BNN summary table.")
    parser.add_argument("--results-dir", type=str, default=str(bnn_toy1d_runs_dir()), help="Root directory with Toy1D run outputs.")
    parser.add_argument("--seed", type=int, default=1, help="Seed identifier used in run paths.")
    parser.add_argument("--samples", type=int, default=300, help="Number of posterior samples used in the run path.")
    parser.add_argument("--thr", type=float, default=1e-5, help="Sparsity threshold (1e-3, 1e-4, 1e-5).")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Training epochs used in run path.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size used in run path.")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE, help="Step size used in run path.")
    parser.add_argument("--outdir", type=str, default=str(bnn_tables_dir()), help="Output directory for CSV and LaTeX files.")
    parser.add_argument("--round-csv", type=int, default=6, help="Rounding precision for CSV outputs.")
    parser.add_argument("--digits-tex", type=int, default=3, help="Digits in LaTeX table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    thr = float(args.thr)
    tag = thr_to_tag(thr)

    rows = []
    missing = []

    for model in MODEL_LIST:
        for nodes in NODES_LIST:
            path = expected_run_path(
                results_dir,
                model=model,
                n_train=int(args.samples),
                nodes=nodes,
                seed=int(args.seed),
                num_epochs=int(args.num_epochs),
                batch_size=int(args.batch_size),
                step_size=float(args.step_size),
            )

            row = {
                "model": pretty_model_name(model),
                "nodes": int(nodes),
                "n_train": int(args.samples),
                "thr": float(thr),
                "thr_tag": tag,
                "std_aleatoric": np.nan,
                "std_epistemic": np.nan,
                "sparsity_layer1": np.nan,
                "sparsity_layer2": np.nan,
            }

            if not path.exists():
                missing.append(str(path))
                rows.append(row)
                continue

            obj = load_npz(path)
            row["std_aleatoric"] = get_std_aleatoric(obj)
            row["std_epistemic"] = get_std_epistemic(obj)
            row["sparsity_layer1"], row["sparsity_layer2"] = get_sparsity_finals(obj, thr=thr)
            rows.append(row)

    df_long = pd.DataFrame(rows).sort_values(["nodes", "model"]).reset_index(drop=True)

    models_tex = [pretty_model_name(m) for m in MODEL_LIST]
    metrics_keys = [k for k, _, _ in METRIC_SPECS]
    columns = pd.MultiIndex.from_product([NODES_LIST, models_tex], names=["Nodes", "Model"])
    wide = pd.DataFrame(index=metrics_keys, columns=columns, dtype=float)

    for nodes in NODES_LIST:
        for model in models_tex:
            sub = df_long[(df_long["nodes"] == nodes) & (df_long["model"] == model)]
            if sub.shape[0] != 1:
                continue
            row = sub.iloc[0]
            for metric_key in metrics_keys:
                wide.loc[metric_key, (nodes, model)] = float(row[metric_key])

    csv_wide = outdir / f"toy1d_results_n={args.samples}_seed={args.seed}_thr={tag}_wide.csv"
    wide.round(int(args.round_csv)).to_csv(csv_wide)

    caption = rf"Results (posterior samples $S={int(args.samples)}$, sparsity threshold $\epsilon={thr_to_tex(thr)}$)."
    tex = to_latex_table_toy1d(
        wide,
        models_tex=models_tex,
        nodes_list=NODES_LIST,
        caption=caption,
        label="tab:toy1d_summary",
        digits=int(args.digits_tex),
    )
    tex_path = outdir / f"toy1d_results_n={args.samples}_seed={args.seed}_thr={tag}.tex"
    tex_path.write_text(tex, encoding="utf-8")

    print(f"Saved: {csv_wide}")
    print(f"Saved: {tex_path}")

    if missing:
        print("\nMissing runs (filled with NaN):")
        for item in missing[:20]:
            print("  -", item)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
