from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import invgamma

from sou.models.horseshoe_estimators import HorseshoeNormalMeans
from sou.models.sou_estimators import SoUNormalMeans
from sou.paths import normal_means_runs_dir


# Exact normal-means experiment used in the paper.
SEED = 1
N_NONZERO = 20
N_TOTAL = 100
BETA_NONZERO = 10.0
SIGMA = 1.0
N_REPLICATES = 5
N_MC_KAPPA = 4000

GAMMA_BETA = 1.0
GAMMA_TAU = 1e-9
GAMMA_NU = 1.0
GAMMA_SIGMA = 1.0
A_PI_SIGMA = 3.0
B_PI_SIGMA = 2.0
LAMBDA_TAU = 10.0
LAMBDA_NU = 1000.0
TOL = 1e-3
MAX_ITER = 10_000

HS_TAU_SCALE = 1.0
HS_LAMBDA_SCALE = 1.0
HS_SIGMA_ALPHA = 3.0
HS_SIGMA_BETA = 2.0
HS_N_ITER = 10_000
HS_N_CHAINS = 4
HS_STAN_FILE = "horseshoe_normal_means_noncentered.stan"


def simulate_data() -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(SEED)
    beta_true = np.concatenate(
        [np.full(N_NONZERO, BETA_NONZERO), np.zeros(N_TOTAL - N_NONZERO)]
    )
    y = [rng.normal(loc=b, scale=SIGMA, size=N_REPLICATES) for b in beta_true]
    return y, beta_true


def sample_sou_kappa(model: SoUNormalMeans) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    nu2 = invgamma(a=model.a_nu_, scale=model.b_nu_).rvs(
        N_MC_KAPPA, random_state=rng
    )
    sigma2 = invgamma(a=model.a_sigma_, scale=model.b_sigma_).rvs(
        N_MC_KAPPA, random_state=rng
    )

    samples = []
    for i in range(N_TOTAL):
        tau2 = invgamma(a=model.a_tau_[i], scale=model.b_tau_[i]).rvs(
            N_MC_KAPPA, random_state=rng
        )
        samples.append(1.0 / (1.0 + N_REPLICATES * tau2 * nu2 / sigma2))

    signal = np.concatenate(samples[:N_NONZERO])
    null = np.concatenate(samples[N_NONZERO:])
    return signal, null


def sample_hs_kappa(model: HorseshoeNormalMeans) -> tuple[np.ndarray, np.ndarray]:
    lambda_draws = np.asarray(model.lambda_)
    tau_draws = np.asarray(model.tau_)
    sigma_draws = np.asarray(model.sigma_)

    samples = []
    for i in range(N_TOTAL):
        local_global2 = (lambda_draws[:, i] * tau_draws) ** 2
        samples.append(
            1.0 / (1.0 + N_REPLICATES * local_global2 / (sigma_draws**2))
        )

    signal = np.concatenate(samples[:N_NONZERO])
    null = np.concatenate(samples[N_NONZERO:])
    return signal, null


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce the normal-means shrinkage experiment from the paper."
    )
    p.add_argument("--outdir", default=str(normal_means_runs_dir()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    y, beta_true = simulate_data()

    hs = HorseshoeNormalMeans(
        tau_scale=HS_TAU_SCALE,
        lambda_scale=HS_LAMBDA_SCALE,
        sigma_alpha=HS_SIGMA_ALPHA,
        sigma_beta=HS_SIGMA_BETA,
        n_iter=HS_N_ITER,
        n_chains=HS_N_CHAINS,
        stan_file=HS_STAN_FILE,
        seed=SEED,
    )
    hs.fit(y)

    sou = SoUNormalMeans(
        gamma_beta=GAMMA_BETA,
        gamma_tau=GAMMA_TAU,
        gamma_nu=GAMMA_NU,
        gamma_sigma=GAMMA_SIGMA,
        a_pi_sigma=A_PI_SIGMA,
        b_pi_sigma=B_PI_SIGMA,
        lambda_tau=LAMBDA_TAU,
        lambda_nu=LAMBDA_NU,
        tol=TOL,
        max_iter=MAX_ITER,
        verbose=False,
    )
    sou.fit(y)

    kappa_sou_nonzero, kappa_sou_zero = sample_sou_kappa(sou)
    kappa_hs_nonzero, kappa_hs_zero = sample_hs_kappa(hs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "kappa_samples.npz"
    json_path = outdir / "kappa_samples.json"

    np.savez_compressed(
        npz_path,
        kappa_sou_nonzero=kappa_sou_nonzero,
        kappa_sou_zero=kappa_sou_zero,
        kappa_hs_nonzero=kappa_hs_nonzero,
        kappa_hs_zero=kappa_hs_zero,
        beta_true=beta_true,
        coef_sou=np.asarray(sou.m_),
        n_nonzero=np.asarray(N_NONZERO),
        n_zero=np.asarray(N_TOTAL - N_NONZERO),
        n_total=np.asarray(N_TOTAL),
        n_replicates=np.asarray(N_REPLICATES),
        seed=np.asarray(SEED),
    )

    meta = {
        "data": {
            "seed": SEED,
            "n_total": N_TOTAL,
            "n_nonzero": N_NONZERO,
            "beta_nonzero": BETA_NONZERO,
            "sigma": SIGMA,
            "n_replicates": N_REPLICATES,
        },
        "SoU-SGL": {
            "gamma_beta": GAMMA_BETA,
            "gamma_tau": GAMMA_TAU,
            "gamma_nu": GAMMA_NU,
            "gamma_sigma": GAMMA_SIGMA,
            "a_pi_sigma": A_PI_SIGMA,
            "b_pi_sigma": B_PI_SIGMA,
            "lambda_tau": LAMBDA_TAU,
            "lambda_nu": LAMBDA_NU,
            "tol": TOL,
            "max_iter": MAX_ITER,
        },
        "Horseshoe": {
            "tau_scale": HS_TAU_SCALE,
            "lambda_scale": HS_LAMBDA_SCALE,
            "sigma_alpha": HS_SIGMA_ALPHA,
            "sigma_beta": HS_SIGMA_BETA,
            "n_iter": HS_N_ITER,
            "n_chains": HS_N_CHAINS,
            "stan_file": HS_STAN_FILE,
        },
        "kappa": "1 / (1 + n_replicates * tau^2 * nu^2 / sigma^2)",
    }
    json_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
