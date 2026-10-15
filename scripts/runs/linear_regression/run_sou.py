from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sou.experiments.linear_regression import (
    INITIAL_GLOBAL_PRECISION,
    INITIAL_LOCAL_PRECISION,
    PAPER_N_SIM,
    PAPER_NOISE_SD,
    PAPER_SEED_BASE,
    dataset_dir,
    fit_paper_sou_regression,
    load_sim_npz_regression,
    normalize_range,
    normalize_scenario,
    prop_small_on_true_zeros,
    result_dir,
    rmse,
)
from sou.models.sou_estimators import SoUGlobalLocalRegressor
from sou.paths import linear_regression_runs_dir


# Exact SoU-SGL specification used for all linear-regression experiments.
GAMMA_BETA = 1.0
GAMMA_TAU = 1e-9
GAMMA_NU = 1.0
GAMMA_SIGMA = 1.0
LAMBDA_TAU = 10.0
LAMBDA_NU = 1000.0
A_PI_SIGMA = 3.0
B_PI_SIGMA = 2.0
TOL = 1e-3
MAX_ITER = 10_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the SoU-SGL linear-regression specification used in the paper."
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
    p.add_argument("--save-latents", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end = normalize_range(args.start, args.end)
    scenario = normalize_scenario(
        args.experiment, args.spec, args.rep, args.prop_nonzero
    )

    data_root = Path(args.data_root)
    ddir = dataset_dir(data_root, scenario)
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
    prop_01 = np.empty(n_chunk)
    prop_001 = np.empty(n_chunk)
    prop_0001 = np.empty(n_chunk)
    iterations = np.empty(n_chunk, dtype=int)
    converged = np.empty(n_chunk, dtype=bool)
    coefs = np.empty((n_chunk, p)) if args.save_coef else None

    latents = (
        {"a_tau": [], "b_tau": [], "a_nu": [], "b_nu": [], "a_sigma": [], "b_sigma": []}
        if args.save_latents
        else None
    )

    for out_idx, sim_idx in enumerate(range(args.start, end)):
        sim_path = ddir / f"sim_{sim_idx:04d}.npz"
        X_train, X_test, y_train, y_test, _, beta_zero_mask = load_sim_npz_regression(
            sim_path, y_as_2d=False
        )

        model = SoUGlobalLocalRegressor(
            fit_intercept=False,
            gamma_beta=GAMMA_BETA,
            gamma_tau=GAMMA_TAU,
            gamma_nu=GAMMA_NU,
            gamma_sigma=GAMMA_SIGMA,
            gamma_alpha=1.0,
            a_pi_sigma=A_PI_SIGMA,
            b_pi_sigma=B_PI_SIGMA,
            lambda_tau=LAMBDA_TAU,
            lambda_nu=LAMBDA_NU,
            tol=TOL,
            max_iter=MAX_ITER,
            verbose=False,
        )
        fit_paper_sou_regression(model, X_train, y_train)

        y_pred = np.asarray(model.predict(X_test)).reshape(-1)
        coef = np.asarray(model.coef_).reshape(-1)
        rmse_arr[out_idx] = rmse(y_pred, y_test)
        prop_01[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-1)
        prop_001[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-2)
        prop_0001[out_idx] = prop_small_on_true_zeros(coef, beta_zero_mask, 1e-3)
        iterations[out_idx] = model.n_iter_
        converged[out_idx] = model.converged_

        if coefs is not None:
            coefs[out_idx] = coef
        if latents is not None:
            latents["a_tau"].append(np.asarray(model.a_tau_))
            latents["b_tau"].append(np.asarray(model.b_tau_))
            for key in ("a_nu", "b_nu", "a_sigma", "b_sigma"):
                latents[key].append(float(getattr(model, f"{key}_")))

    outdir = result_dir(Path(args.outdir), "sou", scenario)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"runs_{args.start:04d}_{end:04d}"
    npz_path = outdir / f"{stem}.npz"
    meta_path = outdir / f"{stem}.json"

    payload = {
        "rmse": rmse_arr,
        "prop_abs_le_0.1": prop_01,
        "prop_abs_le_0.01": prop_001,
        "prop_abs_le_0.001": prop_0001,
        "iterations": iterations,
        "converged": converged,
    }
    if coefs is not None:
        payload["coef"] = coefs
    if latents is not None:
        payload.update({
            "a_tau": np.array(latents["a_tau"], dtype=object),
            "b_tau": np.array(latents["b_tau"], dtype=object),
            "a_nu": np.asarray(latents["a_nu"]),
            "b_nu": np.asarray(latents["b_nu"]),
            "a_sigma": np.asarray(latents["a_sigma"]),
            "b_sigma": np.asarray(latents["b_sigma"]),
        })
    np.savez_compressed(npz_path, **payload)

    meta = {
        "model": "SoU-SGL",
        "experiment": scenario.experiment,
        "spec": scenario.spec,
        "rep": scenario.rep,
        "prop_nonzero": scenario.prop_nonzero,
        "paper_repetitions": PAPER_N_SIM,
        "seed_base": PAPER_SEED_BASE,
        "noise_sd": PAPER_NOISE_SD,
        "range": [args.start, end],
        "configuration": {
            "gamma_beta": GAMMA_BETA,
            "gamma_tau": GAMMA_TAU,
            "gamma_nu": GAMMA_NU,
            "gamma_sigma": GAMMA_SIGMA,
            "lambda_tau": LAMBDA_TAU,
            "lambda_nu": LAMBDA_NU,
            "a_pi_sigma": A_PI_SIGMA,
            "b_pi_sigma": B_PI_SIGMA,
            "initial_local_precision": INITIAL_LOCAL_PRECISION,
            "initial_global_precision": INITIAL_GLOBAL_PRECISION,
            "tol": TOL,
            "max_iter": MAX_ITER,
            "fit_intercept": False,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {npz_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
