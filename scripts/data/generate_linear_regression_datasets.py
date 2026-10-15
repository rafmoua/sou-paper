from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from sou.data.linear_regression import generate_experiment
from sou.experiments.linear_regression import (
    PAPER_N_SIM,
    PAPER_NOISE_SD,
    PAPER_REPS_BY_SPEC,
    PAPER_SEED_BASE,
)


PAPER_PROP_NONZERO = [0.1, 0.2, 0.3, 0.4, 0.5]


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def scenario_dir(
    out_root: Path,
    experiment: int,
    spec: Optional[int],
    rep: int,
    prop_nonzero: Optional[float],
) -> Path:
    spec_str = "none" if spec is None else str(spec)
    out = (
        out_root
        / f"experiment={experiment}"
        / f"spec={spec_str}"
        / f"rep={rep}"
        / f"noise={PAPER_NOISE_SD}"
        / f"seedbase={PAPER_SEED_BASE}"
    )
    if experiment == 3:
        out = out / f"prop={prop_nonzero}"
    return out


def iter_paper_scenarios(
    experiments: List[int], specs: List[int], prop_values: List[float]
) -> Iterable[Tuple[int, Optional[int], int, Optional[float]]]:
    for experiment in experiments:
        if experiment in (1, 2):
            for spec in specs:
                if spec not in PAPER_REPS_BY_SPEC:
                    raise ValueError(f"Unknown spec={spec}")
                for rep in PAPER_REPS_BY_SPEC[spec]:
                    yield experiment, spec, rep, None
        elif experiment == 3:
            for prop in prop_values:
                yield 3, None, 1, float(prop)
        else:
            raise ValueError("experiments must be selected from {1,2,3}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the linear-regression datasets used in the paper."
    )
    p.add_argument("--out-root", default="data/processed/regression")
    p.add_argument(
        "--experiments",
        default="1,2,3",
        help="Subset of paper experiments to generate (default: 1,2,3).",
    )
    p.add_argument(
        "--specs",
        default="0,1,2,3,4",
        help="Subset of specifications for Experiments 1 and 2.",
    )
    p.add_argument(
        "--prop-values",
        default="0.1,0.2,0.3,0.4,0.5",
        help="Subset of signal proportions for Experiment 3.",
    )
    p.add_argument("--start", type=int, default=0, help="First repetition (inclusive).")
    p.add_argument("--end", type=int, default=None, help="Last repetition (exclusive).")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiments = _parse_ints(args.experiments)
    specs = _parse_ints(args.specs)
    prop_values = _parse_floats(args.prop_values)

    end = PAPER_N_SIM if args.end is None else int(args.end)
    if not (0 <= args.start < end <= PAPER_N_SIM):
        raise ValueError(
            f"Invalid range [{args.start}, {end}); the paper uses {PAPER_N_SIM} repetitions"
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if args.dtype == "float32" else np.float64

    for experiment, spec, rep, prop in iter_paper_scenarios(
        experiments, specs, prop_values
    ):
        outdir = scenario_dir(out_root, experiment, spec, rep, prop)
        outdir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "experiment": experiment,
            "spec": spec,
            "rep": rep,
            "prop_nonzero": prop,
            "n_sim": PAPER_N_SIM,
            "seed_base": PAPER_SEED_BASE,
            "noise_sd": PAPER_NOISE_SD,
            "preprocessing": "none",
            "note": "X is generated with i.i.d. standard-normal entries.",
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        for t in range(args.start, end):
            fpath = outdir / f"sim_{t:04d}.npz"
            if fpath.exists() and not args.overwrite:
                continue

            rng = np.random.default_rng(PAPER_SEED_BASE + t)
            X_train, y_train, X_test, y_test, beta_true, meta = generate_experiment(
                experiment,
                spec=spec,
                rep=rep,
                prop_nonzero=prop,
                noise_sd=PAPER_NOISE_SD,
                rng=rng,
            )

            X_train = np.asarray(X_train, dtype=dtype)
            X_test = np.asarray(X_test, dtype=dtype)
            y_train = np.asarray(y_train, dtype=dtype).reshape(-1, 1)
            y_test = np.asarray(y_test, dtype=dtype).reshape(-1, 1)
            beta_true = np.asarray(beta_true, dtype=dtype).reshape(-1)
            beta_nonzero_mask = (beta_true != 0.0).astype(np.uint8)

            np.savez_compressed(
                fpath,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                beta_true=beta_true,
                beta_nonzero_mask=beta_nonzero_mask,
                meta_json=np.asarray(json.dumps(asdict(meta))),
                experiment=np.asarray(experiment),
                spec=np.asarray(-1 if spec is None else spec),
                rep=np.asarray(rep),
                seed=np.asarray(PAPER_SEED_BASE + t),
                prop_nonzero=np.asarray(np.nan if prop is None else prop, dtype=dtype),
            )

        print(f"[OK] {outdir}")


if __name__ == "__main__":
    main()
