from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sou.experiments.linear_regression import PAPER_REPS_BY_SPEC
from sou.paths import linear_regression_runs_dir, linear_regression_tables_dir


MODELS = ["lasso", "hs", "sou"]
MODEL_LABELS = {"lasso": "Lasso", "hs": "HS", "sou": "SoU"}
METRICS = [
    ("rmse", "RMSE", "min"),
    ("prop_abs_le_0.1", r"prop $<10^{-1}$", "max"),
    ("prop_abs_le_0.01", r"prop $<10^{-2}$", "max"),
    ("prop_abs_le_0.001", r"prop $<10^{-3}$", "max"),
]

EXPERIMENT_GRIDS = {
    1: {
        0: (25, 20),
        1: (55, 50),
        2: (105, 100),
        3: (205, 200),
        4: (405, 400),
    },
    2: {
        0: (24, 20),
        1: (60, 50),
        2: (120, 100),
        3: (240, 200),
        4: (480, 400),
    },
}

CAPTIONS = {
    1: (
        "Linear regression simulation: Experiment 1. Test-prediction RMSE and "
        "sparsity among null coefficients, averaged over simulation runs."
    ),
    2: (
        "Linear regression simulation: Experiment 2. Test-prediction RMSE and "
        "sparsity among null coefficients, averaged over simulation runs."
    ),
    3: (
        "Linear regression simulation: Experiment 3 (heavy-tailed signals). "
        "Test-prediction RMSE and sparsity among null coefficients, averaged over simulation runs."
    ),
}


def find_run_npzs(
    rawdir: Path,
    model: str,
    experiment: int,
    spec: int | None,
    rep: int,
    prop_nonzero: float | None,
) -> List[Path]:
    base = (
        rawdir
        / f"model={model}"
        / f"experiment={experiment}"
        / f"spec={-1 if spec is None else spec}"
        / f"rep={rep}"
    )
    if experiment == 3:
        base = base / f"prop={prop_nonzero}"
    return sorted(base.glob("**/runs_*.npz"))


def mean_metric(files: List[Path], key: str) -> float:
    values = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        if key not in data:
            raise KeyError(f"Missing {key!r} in {path}")
        values.append(np.asarray(data[key], dtype=float).reshape(-1))
    if not values:
        return float("nan")
    x = np.concatenate(values)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def build_experiment_12(rawdir: Path, experiment: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for spec, (n, p) in EXPERIMENT_GRIDS[experiment].items():
        for rep in PAPER_REPS_BY_SPEC[spec]:
            row: Dict[str, float] = {"n": n, "p": p, "s": 10 * rep}
            for model in MODELS:
                files = find_run_npzs(rawdir, model, experiment, spec, rep, None)
                for key, _, _ in METRICS:
                    row[f"{key}__{model}"] = mean_metric(files, key)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "p", "s"]).reset_index(drop=True)


def build_experiment_3(rawdir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for prop in [0.1, 0.2, 0.3, 0.4, 0.5]:
        row: Dict[str, float] = {"prop_nonzero": prop}
        for model in MODELS:
            files = find_run_npzs(rawdir, model, 3, None, 1, prop)
            for key, _, _ in METRICS:
                row[f"{key}__{model}"] = mean_metric(files, key)
        rows.append(row)
    return pd.DataFrame(rows)


def _best(values: List[float], mode: str) -> List[bool]:
    x = np.asarray(values, dtype=float)
    if np.all(~np.isfinite(x)):
        return [False] * len(values)
    target = np.nanmin(x) if mode == "min" else np.nanmax(x)
    return [np.isfinite(v) and np.isclose(v, target) for v in values]


def _fmt(value: float, bold: bool) -> str:
    if not np.isfinite(value):
        return r"{\text{--}}"
    text = f"{value:.2f}"
    return rf"\textbf{{{text}}}" if bold else text


def latex_experiment_12(df: pd.DataFrame, experiment: int) -> str:
    n_models = len(MODELS)
    colspec = "@{} rr r " + " ".join(
        [f"*{{{n_models}}}{{S[table-format=1.2]}}" for _ in METRICS]
    ) + " @{}"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{CAPTIONS[experiment]}}}",
        rf"\label{{tab:linear_experiment_{experiment}}}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    groups = " & ".join(
        rf"\multicolumn{{{n_models}}}{{c}}{{{label}}}" for _, label, _ in METRICS
    )
    lines.append(
        r"\multirow{2}{*}{$n$} & \multirow{2}{*}{$p$} & \multirow{2}{*}{$s$} & "
        + groups
        + r" \\"
    )

    start = 4
    lines.append(
        "".join(
            rf"\cmidrule(lr){{{start + i*n_models}-{start + (i+1)*n_models - 1}}}"
            for i in range(len(METRICS))
        )
    )
    model_header = " & ".join(MODEL_LABELS[m] for _ in METRICS for m in MODELS)
    lines.extend([r"& & & " + model_header + r" \\", r"\midrule"])

    for (n, p), group in df.groupby(["n", "p"], sort=True):
        group = group.sort_values("s").reset_index(drop=True)
        for i, row in group.iterrows():
            parts = [f"{int(n)} & {int(p)} & {int(row['s'])}" if i == 0 else f"& & {int(row['s'])}"]
            for key, _, mode in METRICS:
                vals = [float(row[f"{key}__{m}"]) for m in MODELS]
                mask = _best(vals, mode)
                parts.extend(_fmt(v, b) for v, b in zip(vals, mask))
            lines.append(" & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def latex_experiment_3(df: pd.DataFrame) -> str:
    n_models = len(MODELS)
    colspec = "@{} l " + " ".join(
        [f"*{{{n_models}}}{{S[table-format=1.2]}}" for _ in METRICS]
    ) + " @{}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{CAPTIONS[3]}}}",
        r"\label{tab:linear_experiment_3}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    groups = " & ".join(
        rf"\multicolumn{{{n_models}}}{{c}}{{{label}}}" for _, label, _ in METRICS
    )
    lines.append(r"\multirow{2}{*}{prop. signals} & " + groups + r" \\")
    start = 2
    lines.append(
        "".join(
            rf"\cmidrule(lr){{{start + i*n_models}-{start + (i+1)*n_models - 1}}}"
            for i in range(len(METRICS))
        )
    )
    lines.extend([
        " & " + " & ".join(MODEL_LABELS[m] for _ in METRICS for m in MODELS) + r" \\",
        r"\midrule",
    ])
    for _, row in df.iterrows():
        parts = [f"{row['prop_nonzero']:.1f}"]
        for key, _, mode in METRICS:
            vals = [float(row[f"{key}__{m}"]) for m in MODELS]
            mask = _best(vals, mode)
            parts.extend(_fmt(v, b) for v, b in zip(vals, mask))
        lines.append(" & ".join(parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate linear-regression summary tables.")
    p.add_argument("--rawdir", default=str(linear_regression_runs_dir()))
    p.add_argument("--outdir", default=str(linear_regression_tables_dir()))
    p.add_argument("--experiments", default="1,2,3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rawdir = Path(args.rawdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    experiments = [int(x.strip()) for x in args.experiments.split(",") if x.strip()]

    for experiment in experiments:
        if experiment in (1, 2):
            df = build_experiment_12(rawdir, experiment)
            tex = latex_experiment_12(df, experiment)
        elif experiment == 3:
            df = build_experiment_3(rawdir)
            tex = latex_experiment_3(df)
        else:
            raise ValueError("experiments must be selected from {1,2,3}")

        csv_path = outdir / f"experiment_{experiment}_wide.csv"
        tex_path = outdir / f"experiment_{experiment}.tex"
        df.to_csv(csv_path, index=False)
        tex_path.write_text(tex, encoding="utf-8")
        print(f"Saved: {csv_path}")
        print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
