from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sou.paths import bnn_mnist_runs_dir, bnn_tables_dir


DEFAULT_LAYER1_KEY = "fc1"
DEFAULT_LAYER2_KEY = "fc_hidden.0"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_lr(step_size: float) -> str:
    return f"{float(step_size):.2e}"


def _variant_json_path(
    *,
    results_dir: Path,
    variant: str,
    seed: int,
    num_layers: int,
    num_nodes: int,
    hshoe_scale: float,
    num_epochs: int,
    batch_size: int,
    step_size: float,
) -> Path:
    return results_dir / (
        f"mnist_bnn_variant={variant}"
        f"_layers={num_layers}"
        f"_nodes={num_nodes}"
        f"_hs={hshoe_scale}"
        f"_ep={num_epochs}"
        f"_bs={batch_size}"
        f"_lr={_fmt_lr(step_size)}"
        f"_seed={seed}.json"
    )


def list_available_nodes(
    *,
    results_dir: Path,
    variant: str,
    seed: int,
    num_layers: int,
    hshoe_scale: float,
    num_epochs: int,
    batch_size: int,
    step_size: float,
) -> List[int]:
    pattern = (
        f"mnist_bnn_variant={variant}_layers={num_layers}_nodes=*"
        f"_hs={hshoe_scale}_ep={num_epochs}_bs={batch_size}_lr={_fmt_lr(step_size)}_seed={seed}.json"
    )
    nodes: List[int] = []
    for path in results_dir.glob(pattern):
        name = path.name
        i = name.find("_nodes=")
        j = name.find("_hs=", i)
        if i < 0 or j < 0:
            continue
        try:
            nodes.append(int(name[i + len("_nodes="):j]))
        except ValueError:
            continue
    return sorted(set(nodes))


def build_long_table(
    *,
    results_dir: Path,
    variant: str,
    seed: int,
    num_layers: int,
    hshoe_scale: float,
    num_epochs: int,
    batch_size: int,
    step_size: float,
    threshold: float,
    layer1_key: str,
    layer2_key: str,
) -> pd.DataFrame:
    nodes_list = list_available_nodes(
        results_dir=results_dir,
        variant=variant,
        seed=seed,
        num_layers=num_layers,
        hshoe_scale=hshoe_scale,
        num_epochs=num_epochs,
        batch_size=batch_size,
        step_size=step_size,
    )
    if not nodes_list:
        raise RuntimeError("No node configurations found for the selected spec.")

    rows: List[Dict[str, Any]] = []
    for nodes in nodes_list:
        path = _variant_json_path(
            results_dir=results_dir,
            variant=variant,
            seed=seed,
            num_layers=num_layers,
            num_nodes=nodes,
            hshoe_scale=hshoe_scale,
            num_epochs=num_epochs,
            batch_size=batch_size,
            step_size=step_size,
        )
        if not path.exists():
            continue

        run = _read_json(path)
        hist = run.get("history", {})
        nodes_scales = hist.get("nodes_scales", [])
        if not nodes_scales:
            continue

        snap = nodes_scales[-1]
        if layer1_key not in snap or layer2_key not in snap:
            raise KeyError(f"Missing layer keys in {path}: expected {layer1_key!r} and {layer2_key!r}")

        arr1 = np.asarray(snap[layer1_key], dtype=float)
        arr2 = np.asarray(snap[layer2_key], dtype=float)
        accuracy = hist.get("accuracy", [])
        acc_last = float(accuracy[-1]) if accuracy and accuracy[-1] is not None else np.nan

        rows.append(
            {
                "nodes_per_layer": int(nodes),
                "layer1_active": int((arr1 >= threshold).sum()),
                "layer2_active": int((arr2 >= threshold).sum()),
                "test_accuracy_last": acc_last,
            }
        )

    if not rows:
        raise RuntimeError("No rows produced. Check run files and selected parameters.")
    return pd.DataFrame(rows).sort_values("nodes_per_layer").reset_index(drop=True)


def build_wide_table(long_table: pd.DataFrame, *, digits: int = 3) -> pd.DataFrame:
    wide = long_table.loc[
        :, ["nodes_per_layer", "layer1_active", "layer2_active", "test_accuracy_last"]
    ].T
    wide.columns = wide.loc["nodes_per_layer"]
    wide = wide.drop(index="nodes_per_layer")
    wide.columns = [int(float(c)) for c in wide.columns]
    wide = wide.rename(
        index={
            "layer1_active": "Layer 1 active",
            "layer2_active": "Layer 2 active",
            "test_accuracy_last": "Accuracy",
        }
    )

    for row_name in ["Layer 1 active", "Layer 2 active"]:
        wide.loc[row_name] = wide.loc[row_name].astype(float).round(0).astype(int)
    wide.loc["Accuracy"] = wide.loc["Accuracy"].astype(float).round(digits)
    return wide


def to_latex_table(wide: pd.DataFrame, *, caption: str, label: str) -> str:
    node_cols = [int(c) for c in wide.columns]
    colspec = "@{} l " + " ".join(["r" for _ in node_cols]) + " @{}"

    lines: List[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Initial width & " + " & ".join(str(c) for c in node_cols) + r" \\",
        r"\midrule",
    ]

    for row_name in ["Layer 1 active", "Layer 2 active", "Accuracy"]:
        vals: List[str] = []
        for c in node_cols:
            value = wide.loc[row_name, c]
            if pd.isna(value):
                vals.append(r"{\text{--}}")
            elif row_name == "Accuracy":
                vals.append(f"{float(value):.3f}")
            else:
                vals.append(f"{int(round(float(value)))}")
        lines.append(f"{row_name} & " + " & ".join(vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MNIST BNN active-node summary table.")
    parser.add_argument("--results-dir", type=str, default=str(bnn_mnist_runs_dir()))
    parser.add_argument("--outdir", type=str, default=str(bnn_tables_dir()))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--variant", type=str, default="sou_sgl", choices=["sou_sgl", "hs"])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hshoe-scale", type=float, default=1e-1)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--layer1-key", type=str, default=DEFAULT_LAYER1_KEY)
    parser.add_argument("--layer2-key", type=str, default=DEFAULT_LAYER2_KEY)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument(
        "--caption",
        type=str,
        default="MNIST active hidden nodes and final test accuracy across network widths.",
    )
    parser.add_argument("--label", type=str, default="tab:mnist_active_nodes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    long_table = build_long_table(
        results_dir=Path(args.results_dir),
        variant=str(args.variant),
        seed=int(args.seed),
        num_layers=int(args.num_layers),
        hshoe_scale=float(args.hshoe_scale),
        num_epochs=int(args.num_epochs),
        batch_size=int(args.batch_size),
        step_size=float(args.step_size),
        threshold=float(args.threshold),
        layer1_key=str(args.layer1_key),
        layer2_key=str(args.layer2_key),
    )
    wide = build_wide_table(long_table, digits=int(args.digits))

    tag = (
        f"mnist_active_nodes_variant={args.variant}"
        f"_ep={int(args.num_epochs)}_bs={int(args.batch_size)}_lr={_fmt_lr(float(args.step_size))}"
        f"_thr={float(args.threshold):.0e}_seed={int(args.seed)}"
    )
    csv_path = outdir / f"{tag}.csv"
    tex_path = outdir / f"{tag}.tex"

    wide.to_csv(csv_path, index=True)
    tex_path.write_text(
        to_latex_table(wide, caption=str(args.caption), label=str(args.label)),
        encoding="utf-8",
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved TEX: {tex_path}")
    print()
    print(wide)


if __name__ == "__main__":
    main()
