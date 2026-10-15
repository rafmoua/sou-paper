from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoCV

from sou.experiments.linear_regression import (
    PAPER_N_SIM,
    PAPER_NOISE_SD,
    PAPER_SEED_BASE,
    dataset_dir,
    load_sim_npz_regression,
    normalize_range,
    normalize_scenario,
    prop_small_on_true_zeros,
    result_dir,
    rmse,
)
from sou.paths import linear_regression_runs_dir


ALPHAS = np.logspace(np.log10(5e-4), np.log10(20.0), 300)
CV_FOLDS = 5
MAX_ITER = 100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Lasso baseline used in the paper's linear-regression experiments."
    )
    p.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--spec", type=int, default=0)
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--prop-nonzero", type=float, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--data-root", default="data/processed/regression")
    p.add_argument("--outdir", default=str(linear_regression_runs_dir()))
    p.add_argument("--save-coef", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end = normalize_range(args.start, args.end)
    scenario = normalize_scenario(
        args.experiment, args.spec, args.rep, args.prop_nonzero
    )
    ddir = dataset_dir(Path(args.data_root), scenario)
    if not ddir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {ddir}\n"
            "Run scripts/data/generate_linear_regression_datasets.py first."
        )

    sim0 = ddir / f"sim_{args.start:04d}.npz"
    X0, _, _, _, _, _ = load_sim_npz_regression(sim0, y_as_2d=False)
    p = X0.shape[1]
    n_chunk = end - args.start

    rmse_arr = np.empty(n_chunk)
    alpha_arr = np.empty(n_chunk)
    prop_01 = np.empty(n_chunk)
    prop_001 = np.empty(n_chunk)
    prop_0001 = np.empty(n_chunk)
    coefs = np.empty((n_chunk, p)) if args.save_coef else None

    for out_idx, sim_idx in enumerate(range(args.start, end)):
        sim_path = ddir / f"sim_{sim_idx:04d}.npz"
        X_train, X_test, y_train, y_test, _, beta_zero_mask = load_sim_npz_regression(
            sim_path, y_as_2d=False
        )
        model = LassoCV(
            alphas=ALPHAS,
            fit_intercept=False,
            cv=CV_FOLDS,
            max_iter=MAX_ITER,
        )
        model.fit(X_train, y_train)

        coef = np.asarray(model.coef_).reshape(-1)
        rmse_arr[out_idx] = rmse(model.predict(X_test), y_test)
        alpha_arr[out_idx] = float(model.alpha_)
        prop_01[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-1)
        prop_001[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-2)
        prop_0001[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-3)
        if coefs is not None:
            coefs[out_idx] = coef

    outdir = result_dir(Path(args.outdir), "lasso", scenario)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"runs_{args.start:04d}_{end:04d}"
    npz_path = outdir / f"{stem}.npz"
    meta_path = outdir / f"{stem}.json"

    payload = {
        "rmse": rmse_arr,
        "alpha": alpha_arr,
        "prop_abs_le_0.1": prop_01,
        "prop_abs_le_0.01": prop_001,
        "prop_abs_le_0.001": prop_0001,
    }
    if coefs is not None:
        payload["coef"] = coefs
    np.savez_compressed(npz_path, **payload)

    meta = {
        "model": "Lasso",
        "experiment": scenario.experiment,
        "spec": scenario.spec,
        "rep": scenario.rep,
        "prop_nonzero": scenario.prop_nonzero,
        "paper_repetitions": PAPER_N_SIM,
        "seed_base": PAPER_SEED_BASE,
        "noise_sd": PAPER_NOISE_SD,
        "range": [args.start, end],
        "configuration": {
            "alpha_min": float(ALPHAS.min()),
            "alpha_max": float(ALPHAS.max()),
            "n_alphas": int(ALPHAS.size),
            "alpha_grid": "log",
            "cv_folds": CV_FOLDS,
            "max_iter": MAX_ITER,
            "fit_intercept": False,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {npz_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
