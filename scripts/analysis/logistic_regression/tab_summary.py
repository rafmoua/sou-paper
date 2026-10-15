from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sou.paths import logistic_regression_ovr_runs_dir, logistic_regression_tables_dir


SEED = 1
DIGITS = list(range(10))
CUTS = [1e-1, 1e-2, 1e-3]
MODELS = ["none", "l1cv", "gl", "sou"]
PRETTY = {
    "none": "LR",
    "l1cv": "LR-L1",
    "gl": "GL",
    "sou": "SoU-SGL",
}


def load_digit_file(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def accuracy(per_digit: Dict[int, dict], model: str, y_test: np.ndarray) -> float:
    proba = np.column_stack(
        [np.asarray(per_digit[d][f"proba_test_{model}"]).reshape(-1) for d in DIGITS]
    )
    return float(np.mean(np.argmax(proba, axis=1) == y_test))


def sparsity(per_digit: Dict[int, dict], model: str, cut: float) -> float:
    props = []
    for digit in DIGITS:
        coef = np.asarray(per_digit[digit][f"coef_{model}"]).reshape(-1)
        props.append(float(np.mean(np.abs(coef) < cut)))
    return float(np.mean(props))


def _best(values: List[float]) -> List[bool]:
    x = np.asarray(values, dtype=float)
    target = np.nanmax(x)
    return [np.isfinite(v) and np.isclose(v, target) for v in values]


def to_latex(df: pd.DataFrame) -> str:
    bold_acc = _best(df["accuracy"].tolist())
    bold_prop = {
        cut: _best(df[f"prop_eps_{cut:g}"].tolist()) for cut in CUTS
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MNIST logistic regression: test accuracy and numerical sparsity for the four considered models.}",
        r"\label{tab:logistic_accuracy}",
        r"\footnotesize",
        r"\begin{tabular}{@{} l S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] @{} }",
        r"\toprule",
        r"\multirow{2}{*}{Model} & \multicolumn{1}{c}{\multirow{2}{*}{Accuracy}} & \multicolumn{3}{c}{Sparsity: prop $|\hat\beta_j|<\epsilon$} \\",
        r"\cmidrule(l){3-5}",
        r" & & \multicolumn{1}{c}{$\epsilon=10^{-1}$} & \multicolumn{1}{c}{$\epsilon=10^{-2}$} & \multicolumn{1}{c}{$\epsilon=10^{-3}$} \\",
        r"\midrule",
    ]

    for i, row in df.iterrows():
        acc = f"{row['accuracy']:.3f}"
        if bold_acc[i]:
            acc = rf"\textbf{{{acc}}}"
        vals = []
        for cut in CUTS:
            text = f"{row[f'prop_eps_{cut:g}']:.2f}"
            if bold_prop[cut][i]:
                text = rf"\textbf{{{text}}}"
            vals.append(text)
        label = row["model"] + (" (ours)" if row["model"] == "SoU-SGL" else "")
        lines.append(" & ".join([label, acc, *vals]) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the MNIST logistic-regression table.")
    p.add_argument("--rawdir", default=str(logistic_regression_ovr_runs_dir()))
    p.add_argument("--mnist-cache", default="data/raw/mnist_784_openml_v1.npz")
    p.add_argument("--outdir", default=str(logistic_regression_tables_dir()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rawdir = Path(args.rawdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mnist = np.load(Path(args.mnist_cache), allow_pickle=True)
    y_test = np.asarray(mnist["y"]).astype(int)[60000:]

    per_digit = {}
    for digit in DIGITS:
        path = rawdir / f"mnist_ovr_digit={digit}_seed={SEED}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        per_digit[digit] = load_digit_file(path)

    rows = []
    for model in MODELS:
        row = {
            "model": PRETTY[model],
            "accuracy": accuracy(per_digit, model, y_test),
        }
        for cut in CUTS:
            row[f"prop_eps_{cut:g}"] = sparsity(per_digit, model, cut)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = outdir / "logistic_regression_seed=1_wide.csv"
    tex_path = outdir / "logistic_regression_seed=1.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(to_latex(df), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    main()
