from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def normal_means_runs_dir() -> Path:
    return RESULTS_DIR / "normal_means" / "runs"


def linear_regression_runs_dir() -> Path:
    return RESULTS_DIR / "linear_regression" / "runs"


def linear_regression_tables_dir() -> Path:
    return RESULTS_DIR / "linear_regression" / "tables"


def logistic_regression_ovr_runs_dir() -> Path:
    return RESULTS_DIR / "logistic_regression" / "runs" / "ovr"


def logistic_regression_tables_dir() -> Path:
    return RESULTS_DIR / "logistic_regression" / "tables"


def bnn_mnist_runs_dir() -> Path:
    return RESULTS_DIR / "bnn" / "runs" / "mnist"


def bnn_toy1d_runs_dir() -> Path:
    return RESULTS_DIR / "bnn" / "runs" / "toy1d"


def bnn_tables_dir() -> Path:
    return RESULTS_DIR / "bnn" / "tables"


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
