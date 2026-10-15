from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sou.experiments.linear_regression import PAPER_REPS_BY_SPEC
from sou.paths import (
    bnn_mnist_runs_dir,
    bnn_tables_dir,
    bnn_toy1d_runs_dir,
    linear_regression_runs_dir,
    linear_regression_tables_dir,
    logistic_regression_ovr_runs_dir,
    logistic_regression_tables_dir,
    normal_means_runs_dir,
)


def run(cmd: List[str], dry_run: bool) -> None:
    print("[run]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def py(script: str, *args: str) -> List[str]:
    return [sys.executable, script, *args]


def _paper_linear_scenarios() -> Iterable[tuple[int, int, int, str | None]]:
    for experiment in (1, 2):
        for spec, reps in PAPER_REPS_BY_SPEC.items():
            for rep in reps:
                yield experiment, spec, rep, None
    for prop in ("0.1", "0.2", "0.3", "0.4", "0.5"):
        yield 3, 0, 1, prop


def compile_sources(dry_run: bool) -> None:
    py_files = [str(p) for p in PROJECT_ROOT.joinpath("sou").rglob("*.py")]
    py_files.extend(str(p) for p in PROJECT_ROOT.joinpath("scripts").rglob("*.py"))
    run([sys.executable, "-m", "py_compile", *py_files], dry_run)


def analysis_mode(dry_run: bool) -> None:
    """Regenerate manuscript figures and tables from existing result files."""
    compile_sources(dry_run)
    commands = [
        py("scripts/analysis/one_parameter/fig_illustration.py"),
        py(
            "scripts/analysis/normal_means/fig_kappa_panels.py",
            "--npz",
            str(normal_means_runs_dir() / "kappa_samples.npz"),
        ),
        py(
            "scripts/analysis/linear_regression/tab_experiments_summary.py",
            "--rawdir",
            str(linear_regression_runs_dir()),
            "--outdir",
            str(linear_regression_tables_dir()),
        ),
        py(
            "scripts/analysis/linear_regression/fig_zero_coefficients.py",
            "--results-root",
            str(linear_regression_runs_dir()),
            "--data-root",
            "data/processed/regression",
        ),
        py(
            "scripts/analysis/logistic_regression/tab_summary.py",
            "--rawdir",
            str(logistic_regression_ovr_runs_dir()),
            "--outdir",
            str(logistic_regression_tables_dir()),
        ),
        py(
            "scripts/analysis/logistic_regression/fig_coefficients_heatmap.py",
            "--rawdir",
            str(logistic_regression_ovr_runs_dir()),
            "--seed",
            "1",
        ),
        py(
            "scripts/analysis/bnn/fig_mnist_training_dynamics.py",
            "--results-dir",
            str(bnn_mnist_runs_dir()),
            "--seed",
            "1",
        ),
        py(
            "scripts/analysis/bnn/tab_mnist_active_nodes.py",
            "--results-dir",
            str(bnn_mnist_runs_dir()),
            "--outdir",
            str(bnn_tables_dir()),
            "--seed",
            "1",
        ),
        py(
            "scripts/analysis/bnn/fig_toy1d_prediction_sparsity.py",
            "--results-dir",
            str(bnn_toy1d_runs_dir()),
        ),
        py(
            "scripts/analysis/bnn/tab_toy1d_summary.py",
            "--results-dir",
            str(bnn_toy1d_runs_dir()),
            "--outdir",
            str(bnn_tables_dir()),
            "--samples",
            "300",
            "--seed",
            "1",
            "--thr",
            "1e-5",
        ),
    ]
    for command in commands:
        run(command, dry_run)


def full_mode(dry_run: bool) -> None:
    """Run the complete set of experiments used in the paper, then analyse them."""
    run(py("scripts/data/download_mnist_openml.py"), dry_run)
    run(py("scripts/data/generate_linear_regression_datasets.py"), dry_run)
    run(
        py(
            "scripts/runs/normal_means/run_normal_means.py",
            "--outdir",
            str(normal_means_runs_dir()),
        ),
        dry_run,
    )

    for experiment, spec, rep, prop in _paper_linear_scenarios():
        common = [
            "--experiment",
            str(experiment),
            "--spec",
            str(spec),
            "--rep",
            str(rep),
            "--save-coef",
            "--outdir",
            str(linear_regression_runs_dir()),
        ]
        if experiment == 3:
            common.extend(["--prop-nonzero", str(prop)])
        for script in ("run_lasso.py", "run_hs.py", "run_sou.py"):
            run(py(f"scripts/runs/linear_regression/{script}", *common), dry_run)

    for digit in range(10):
        run(
            py(
                "scripts/runs/logistic_regression/run_mnist_ovr_digit.py",
                "--digit",
                str(digit),
                "--outdir",
                str(logistic_regression_ovr_runs_dir()),
            ),
            dry_run,
        )

    run(
        py(
            "scripts/runs/bnn/run_mnist_bnn.py",
            "--seed",
            "1",
            "--num-nodes",
            "200",
            "--variants",
            "bnn",
            "hs",
            "sou_sgl",
            "--outdir",
            str(bnn_mnist_runs_dir()),
        ),
        dry_run,
    )
    for nodes in (50, 100, 400, 800, 1200, 1600):
        run(
            py(
                "scripts/runs/bnn/run_mnist_bnn.py",
                "--seed",
                "1",
                "--num-nodes",
                str(nodes),
                "--variants",
                "sou_sgl",
                "--outdir",
                str(bnn_mnist_runs_dir()),
            ),
            dry_run,
        )

    for model in ("bnn", "hs", "sou_sgl"):
        for n_train in (100, 150, 300):
            for nodes in (100, 200, 400):
                run(
                    py(
                        "scripts/runs/bnn/run_toy1d_one.py",
                        "--model-label",
                        model,
                        "--n-train",
                        str(n_train),
                        "--nodes",
                        str(nodes),
                        "--seed",
                        "1",
                        "--outdir",
                        str(bnn_toy1d_runs_dir()),
                    ),
                    dry_run,
                )

    analysis_mode(dry_run)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce the numerical results of the SoU paper.")
    p.add_argument(
        "--mode",
        choices=["analysis", "full"],
        default="analysis",
        help="analysis regenerates outputs from saved runs; full reruns every experiment first.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "analysis":
        analysis_mode(args.dry_run)
    else:
        full_mode(args.dry_run)


if __name__ == "__main__":
    main()
