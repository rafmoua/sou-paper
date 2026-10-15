from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from sou.models.BNNs.bnn import SoUBnnRegression  # noqa: F401
from sou.paths import bnn_toy1d_runs_dir


def set_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy, Python and PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def toy_dataset(n: int, noise_std: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic Toy1D train samples from a noisy sine signal."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-4.0, 4.0, size=n)
    y_true = np.sin(X)
    y = y_true + rng.normal(0.0, noise_std, size=n)
    return X[:, None], y, y_true


def make_x_test_grid(n_grid: int = 200, x_min: float = -4.0, x_max: float = 4.0) -> np.ndarray:
    """Create the fixed evaluation grid used for predictions."""
    return np.linspace(x_min, x_max, n_grid)[:, None]


def to_serializable(obj: Any) -> Any:
    """Convert nested objects to JSON-friendly structures when possible."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    return obj


def extract_layer_sparsity_series(
    history: Dict,
    layer_key: str,
    abs_threshold: float,
    source: str = "W_effective",
) -> np.ndarray:
    """Extract one layer sparsity trajectory from training history."""
    out = []
    for ep in history.get("sparsity", []):
        if ep is None:
            out.append(np.nan)
            continue

        if source not in ep or not isinstance(ep[source], dict):
            raise ValueError(f"Expected history['sparsity'][epoch]['{source}'] to be a dict.")
        layer_map = ep[source]

        if layer_key not in layer_map:
            out.append(np.nan)
            continue

        v = layer_map[layer_key]

        # Case A: already a scalar prop
        if isinstance(v, (float, int, np.floating, np.integer)):
            out.append(float(v))
            continue

        # Case B: list-like stored as [thr1e-3, thr1e-4, thr1e-5]
        if isinstance(v, (list, tuple, np.ndarray)):
            # mapping: idx 0->1e-3, idx 1->1e-4, idx 2->1e-5
            if abs_threshold == 1e-3:
                idx = 0
            elif abs_threshold == 1e-4:
                idx = 1
            elif abs_threshold == 1e-5:
                idx = 2
            else:
                # fallback to 1e-4 convention
                idx = 1
            try:
                out.append(float(v[idx]))
            except Exception:
                out.append(np.nan)
            continue

        out.append(np.nan)

    return np.asarray(out, dtype=np.float32)


def std_epistemic_from_mc_samples(y_samples: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute epistemic std curve and mean from MC predictive samples."""
    y_samples = np.asarray(y_samples)
    if y_samples.ndim == 3 and y_samples.shape[-1] == 1:
        y_samples = y_samples[..., 0]
    if y_samples.ndim != 2:
        raise ValueError(f"Unexpected y_samples shape: {y_samples.shape}")

    std_curve = y_samples.std(axis=0)
    return std_curve.astype(np.float32), float(std_curve.mean())


def resolve_model_and_gamma(model_label: str) -> Tuple[str, float]:
    """Map external model label to internal model kind and gamma_local."""
    ml = model_label.lower()
    if ml == "bnn":
        return "bnn", 1.0  # gamma_local unused for bnn, keep as metadata
    if ml == "hs":
        return "sou", 1.0
    if ml in {"sou_sgl", "sou-sgl", "sou"}:
        return "sou", 1e-9
    raise ValueError("model-label must be one of {'bnn','hs','sou_sgl'}")


def _fmt_lr(step_size: float) -> str:
    """Stable compact formatting for learning-rate token in folder names."""
    return f"{float(step_size):.2e}"


def main() -> None:
    p = argparse.ArgumentParser(description="Run one Toy1D BNN configuration and save outputs.")
    p.add_argument("--model-label", type=str, required=True, choices=["bnn", "hs", "sou_sgl"], help="Model variant label.")
    p.add_argument("--n-train", type=int, required=True, choices=[100, 150, 300], help="Number of training samples.")
    p.add_argument("--nodes", type=int, required=True, choices=[100, 200, 400], help="Hidden nodes per layer.")
    p.add_argument("--seed", type=int, default=1, help="Global random seed.")

    p.add_argument("--num-epochs", type=int, default=30000, help="Training epochs.")
    p.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers.")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    p.add_argument("--step-size", type=float, default=0.005, help="Optimizer step size.")
    p.add_argument("--warmup-epochs", type=int, default=50, help="Warmup epochs.")

    p.add_argument("--noise-std", type=float, default=0.2, help="Observation noise std in data generation.")
    p.add_argument("--xgrid", type=int, default=200, help="Number of test-grid points.")
    p.add_argument("--mc-samples", type=int, default=200, help="MC samples for predictive uncertainty.")

    p.add_argument("--checkpoint-epoch", type=int, default=30000, help="Epoch index for checkpoint prediction.")
    p.add_argument("--hshoe-scale", type=float, default=1e-1, help="Horseshoe scale.")

    p.add_argument("--layer1-key", type=str, default="fc1", help="Layer-1 key in history['sparsity'].")
    p.add_argument("--layer2-key", type=str, default="fc_hidden.0", help="Layer-2 key in history['sparsity'].")

    p.add_argument("--outdir", type=str, default=str(bnn_toy1d_runs_dir()), help="Root output directory.")
    p.add_argument("--save-samples", action="store_true", help="Also save MC samples (can be large).")
    p.add_argument("--save-full-history", action="store_true", help="Also save full training history in NPZ/JSON.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose fitting logs.")
    args = p.parse_args()

    set_seed(int(args.seed))

    # ---- Data ----
    X_train_np, y_train_np, _y_true_np = toy_dataset(
        n=int(args.n_train),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )
    X_test_np = make_x_test_grid(n_grid=int(args.xgrid))

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    _X_test = torch.tensor(X_test_np, dtype=torch.float32)  # kept for parity with your previous code

    # ---- Model resolution ----
    model_kind, gamma_local = resolve_model_and_gamma(args.model_label)

    num_layers = int(args.num_layers)
    num_nodes = int(args.nodes)

    config = {
        "num_epochs": int(args.num_epochs),
        "batch_size": int(args.batch_size),
        "verbose": bool(args.verbose),
        "num_nodes": [num_nodes for _ in range(num_layers)],
        "num_layers": num_layers,
        "ip_dim": int(X_train.shape[1]),
        "op_dim": 1,
        "step_size": float(args.step_size),
        "hshoe_scale": [float(args.hshoe_scale) for _ in range(num_layers)],
        "gamma_local": [float(gamma_local) for _ in range(num_layers)],
        "activation_type": "relu",
        "warmup_epochs": int(args.warmup_epochs),
    }

    # ---- Fit ----
    set_seed(int(args.seed))
    model = SoUBnnRegression(config=config, model=model_kind)

    checkpoint_epochs = [int(args.checkpoint_epoch)]
    model.fit(X_train, y_train, checkpoint_epochs=checkpoint_epochs)

    # ---- Predict on grid (MC) ----
    device = getattr(model, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test_t = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    pred_obj = model.predict(X_test_t, checkpoint_epoch=int(args.checkpoint_epoch), mc_samples=int(args.mc_samples))

    # If it returns namedtuple with y_prob_samples / y_pred etc, handle.
    if hasattr(pred_obj, "y_prob_samples"):
        y_samples = np.asarray(pred_obj.y_prob_samples)
    else:
        y_samples = np.asarray(pred_obj)

    # Normalize samples shape to (mc, n_grid)
    if y_samples.ndim == 1:
        y_samples = y_samples[None, :]
    if y_samples.ndim == 2:
        pass
    elif y_samples.ndim == 3 and y_samples.shape[-1] == 1:
        y_samples = y_samples[..., 0]
    else:
        raise ValueError(f"Unexpected prediction sample shape: {y_samples.shape}")

    pred_mean = y_samples.mean(axis=0).astype(np.float32)
    pred_std_curve, std_epistemic_scalar = std_epistemic_from_mc_samples(y_samples)

    # Aleatoric
    noise_var = None
    if hasattr(model, "net") and hasattr(model.net, "get_noise_var"):
        try:
            noise_var = float(model.net.get_noise_var())
        except Exception:
            noise_var = None

    # ---- History -> sparsity series for ALL thresholds ----
    history = getattr(model, "history", None) or {}
    history_serializable = to_serializable(history) if args.save_full_history else None
    neg_elbo = np.asarray(history.get("neg_elbo", []), dtype=np.float32)

    # We always extract these three (matches your training code)
    thresholds = [1e-3, 1e-4, 1e-5]

    sparsity = {
        "thresholds": thresholds,
        "layer_keys": {"layer1": str(args.layer1_key), "layer2": str(args.layer2_key)},
        # W_effective sparsity
        "layer1": {},
        "layer2": {},
        "layer1_final": {},
        "layer2_final": {},
        # Neuron/node sparsity (from history['sparsity'][epoch]['neuron'])
        "node_layer1": {},
        "node_layer2": {},
        "node_layer1_final": {},
        "node_layer2_final": {},
    }

    for thr in thresholds:
        s1 = extract_layer_sparsity_series(
            history=history,
            layer_key=str(args.layer1_key),
            abs_threshold=float(thr),
            source="W_effective",
        )
        s2 = extract_layer_sparsity_series(
            history=history,
            layer_key=str(args.layer2_key),
            abs_threshold=float(thr),
            source="W_effective",
        )

        ns1 = extract_layer_sparsity_series(
            history=history,
            layer_key=str(args.layer1_key),
            abs_threshold=float(thr),
            source="neuron",
        )
        ns2 = extract_layer_sparsity_series(
            history=history,
            layer_key=str(args.layer2_key),
            abs_threshold=float(thr),
            source="neuron",
        )

        # store as lists for JSON
        sparsity["layer1"][str(thr)] = s1.tolist()
        sparsity["layer2"][str(thr)] = s2.tolist()
        sparsity["node_layer1"][str(thr)] = ns1.tolist()
        sparsity["node_layer2"][str(thr)] = ns2.tolist()

        # finals
        s1_final = float(s1[~np.isnan(s1)][-1]) if np.any(~np.isnan(s1)) else np.nan
        s2_final = float(s2[~np.isnan(s2)][-1]) if np.any(~np.isnan(s2)) else np.nan
        ns1_final = float(ns1[~np.isnan(ns1)][-1]) if np.any(~np.isnan(ns1)) else np.nan
        ns2_final = float(ns2[~np.isnan(ns2)][-1]) if np.any(~np.isnan(ns2)) else np.nan
        sparsity["layer1_final"][str(thr)] = s1_final
        sparsity["layer2_final"][str(thr)] = s2_final
        sparsity["node_layer1_final"][str(thr)] = ns1_final
        sparsity["node_layer2_final"][str(thr)] = ns2_final

    # ---- Save ----
    outdir = (
        Path(args.outdir)
        / f"model={args.model_label}"
        / f"n={args.n_train}"
        / f"nodes={args.nodes}"
        / f"ep={args.num_epochs}_bs={args.batch_size}_lr={_fmt_lr(args.step_size)}"
        / f"seed={args.seed}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    npz_path = outdir / "run.npz"
    json_path = outdir / "run.json"

    # NPZ payload: keep arrays as arrays (good for later plotting/aggregation)
    payload = dict(
        # metadata
        model_label=str(args.model_label),
        model_kind=str(model_kind),
        gamma_local=float(gamma_local),
        seed=int(args.seed),
        n_train=int(args.n_train),
        nodes=int(args.nodes),
        num_layers=int(args.num_layers),
        num_epochs=int(args.num_epochs),
        checkpoint_epoch=int(args.checkpoint_epoch),
        noise_std=float(args.noise_std),
        mc_samples=int(args.mc_samples),
        # xgrid + predictions
        x_test=X_test_np.astype(np.float32),
        pred_mean=pred_mean,
        pred_std_epistemic=pred_std_curve,
        std_epistemic_scalar=float(std_epistemic_scalar),
        noise_var=(np.nan if noise_var is None else float(noise_var)),
        # training curve
        neg_elbo=neg_elbo,
    )

    if args.save_full_history:
        # complete training history for post-hoc analyses/animations
        payload["history"] = np.array(history_serializable, dtype=object)

    # Also store sparsity arrays per-threshold in NPZ (named fields)
    for thr in thresholds:
        s1_arr = np.asarray(sparsity["layer1"][str(thr)], dtype=np.float32)
        s2_arr = np.asarray(sparsity["layer2"][str(thr)], dtype=np.float32)
        ns1_arr = np.asarray(sparsity["node_layer1"][str(thr)], dtype=np.float32)
        ns2_arr = np.asarray(sparsity["node_layer2"][str(thr)], dtype=np.float32)
        payload[f"sparsity_layer1_thr_{thr:g}"] = s1_arr
        payload[f"sparsity_layer2_thr_{thr:g}"] = s2_arr
        payload[f"sparsity_layer1_final_thr_{thr:g}"] = np.float32(sparsity["layer1_final"][str(thr)])
        payload[f"sparsity_layer2_final_thr_{thr:g}"] = np.float32(sparsity["layer2_final"][str(thr)])
        payload[f"node_sparsity_layer1_thr_{thr:g}"] = ns1_arr
        payload[f"node_sparsity_layer2_thr_{thr:g}"] = ns2_arr
        payload[f"node_sparsity_layer1_final_thr_{thr:g}"] = np.float32(sparsity["node_layer1_final"][str(thr)])
        payload[f"node_sparsity_layer2_final_thr_{thr:g}"] = np.float32(sparsity["node_layer2_final"][str(thr)])

    if args.save_samples:
        payload["pred_samples"] = y_samples.astype(np.float32)

    np.savez_compressed(npz_path, **payload)

    # JSON: include ALL thresholds (this is what you asked for)
    # Keep it reasonably sized: store sparsity series as lists, but do NOT dump big arrays like pred_samples.
    meta = {
        "model_label": str(args.model_label),
        "model_kind": str(model_kind),
        "gamma_local": float(gamma_local),
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "nodes": int(args.nodes),
        "num_layers": int(args.num_layers),
        "num_epochs": int(args.num_epochs),
        "checkpoint_epoch": int(args.checkpoint_epoch),
        "noise_std": float(args.noise_std),
        "mc_samples": int(args.mc_samples),
        "sparsity": sparsity,  # <-- ALL thresholds here
        "saved_npz_fields": sorted([k for k in payload.keys()]),
    }
    if args.save_full_history:
        meta["history"] = history_serializable
    if args.save_samples:
        meta["note"] = "pred_samples stored in NPZ (not in JSON) because it can be large."

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")
    print(
        "Shapes:",
        "x_test", payload["x_test"].shape,
        "pred_mean", payload["pred_mean"].shape,
        "pred_std_epistemic", payload["pred_std_epistemic"].shape,
        "neg_elbo", payload["neg_elbo"].shape,
    )


if __name__ == "__main__":
    main()
