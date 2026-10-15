from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from sou.experiments.logistic_regression import (
    INITIAL_GLOBAL_PRECISION,
    INITIAL_LOCAL_PRECISION,
    fit_paper_classifier,
)
from sou.models.sou_estimators import SoUGlobalLocalClassifier
from sou.paths import logistic_regression_ovr_runs_dir


SEED = 1
LAMBDA_TAU = 10.0
LAMBDA_NU = 1000.0
SIGMA_ALPHA = 10.0
GAMMA_ALPHA = 1.0
GAMMA_BETA = 1.0
GAMMA_NU = 1.0
GAMMA_TAU_GL = 1.0
GAMMA_TAU_SOU = 1e-6
TOL = 1e-3
MAX_ITER = 10_000


def load_mnist(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.int64)
    y = np.asarray(data["y"]).astype(int)
    if X.shape != (70000, 784) or y.shape != (70000,):
        raise ValueError(f"Unexpected MNIST shapes: X={X.shape}, y={y.shape}")
    return X, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one MNIST one-vs-rest task with the paper specifications."
    )
    p.add_argument("--digit", type=int, required=True, choices=range(10))
    p.add_argument("--mnist-cache", default="data/raw/mnist_784_openml_v1.npz")
    p.add_argument("--outdir", default=str(logistic_regression_ovr_runs_dir()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_mnist(Path(args.mnist_cache))

    X = (X / 255.0).astype(np.float32)
    X_train, X_test = X[:60000], X[60000:]
    y_train_mc = y[:60000]
    y_train = (y_train_mc == args.digit).astype(np.int64)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    common = dict(
        gamma_alpha=GAMMA_ALPHA,
        gamma_beta=GAMMA_BETA,
        gamma_nu=GAMMA_NU,
        sigma_alpha=SIGMA_ALPHA,
        lambda_tau=LAMBDA_TAU,
        lambda_nu=LAMBDA_NU,
        tol=TOL,
        max_iter=MAX_ITER,
        verbose=False,
    )
    sou = SoUGlobalLocalClassifier(gamma_tau=GAMMA_TAU_SOU, **common)
    gl = SoUGlobalLocalClassifier(gamma_tau=GAMMA_TAU_GL, **common)
    fit_paper_classifier(sou, X_train, y_train)
    fit_paper_classifier(gl, X_train, y_train)

    lr_l1 = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        penalty="l1",
        solver="saga",
        tol=1e-4,
        max_iter=MAX_ITER,
        random_state=SEED,
        n_jobs=-1,
    )
    lr = LogisticRegression(
        penalty=None,
        max_iter=MAX_ITER,
        random_state=SEED,
    )
    lr_l1.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / f"mnist_ovr_digit={args.digit}_seed={SEED}.npz"
    out_json = outdir / f"mnist_ovr_digit={args.digit}_seed={SEED}.json"

    np.savez_compressed(
        out_npz,
        digit=np.asarray(args.digit),
        seed=np.asarray(SEED),
        proba_test_sou=np.asarray(sou.predict_proba(X_test), dtype=np.float32),
        proba_test_gl=np.asarray(gl.predict_proba(X_test), dtype=np.float32),
        proba_test_l1cv=np.asarray(lr_l1.predict_proba(X_test)[:, 1], dtype=np.float32),
        proba_test_none=np.asarray(lr.predict_proba(X_test)[:, 1], dtype=np.float32),
        coef_sou=np.asarray(sou.m_, dtype=np.float32),
        coef_gl=np.asarray(gl.m_, dtype=np.float32),
        coef_l1cv=np.asarray(lr_l1.coef_[0], dtype=np.float32),
        coef_none=np.asarray(lr.coef_[0], dtype=np.float32),
        sou_iterations=np.asarray(sou.n_iter_),
        gl_iterations=np.asarray(gl.n_iter_),
        sou_converged=np.asarray(sou.converged_),
        gl_converged=np.asarray(gl.converged_),
    )

    meta = {
        "digit": args.digit,
        "seed": SEED,
        "data": {
            "train": 60000,
            "test": 10000,
            "preprocessing": "pixels divided by 255, then standardised feature-wise using training-set mean and standard deviation",
        },
        "GL": {
            "gamma_tau": GAMMA_TAU_GL,
            "lambda_tau": LAMBDA_TAU,
            "lambda_nu": LAMBDA_NU,
            "sigma_alpha_sd": SIGMA_ALPHA,
            "initial_local_precision": INITIAL_LOCAL_PRECISION,
            "initial_global_precision": INITIAL_GLOBAL_PRECISION,
            "tol": TOL,
            "max_iter": MAX_ITER,
        },
        "SoU-SGL": {
            "gamma_tau": GAMMA_TAU_SOU,
            "lambda_tau": LAMBDA_TAU,
            "lambda_nu": LAMBDA_NU,
            "sigma_alpha_sd": SIGMA_ALPHA,
            "initial_local_precision": INITIAL_LOCAL_PRECISION,
            "initial_global_precision": INITIAL_GLOBAL_PRECISION,
            "tol": TOL,
            "max_iter": MAX_ITER,
        },
        "LR-L1": {
            "penalty": "l1",
            "solver": "saga",
            "Cs": "20 log-spaced values from 1e-3 to 1e2",
            "cv": "sklearn default",
        },
    }
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {out_npz}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
