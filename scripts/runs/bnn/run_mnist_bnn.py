from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sou.paths import bnn_mnist_runs_dir
from sou.models.BNNs.bnn import SoUBnnClassificationHistory


MNIST_CACHE_DEFAULT = "data/raw/mnist_784_openml_v1.npz"
OUTDIR_DEFAULT = str(bnn_mnist_runs_dir())

# Fixed experimental settings
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_NODES = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_STEP_SIZE = 0.001
DEFAULT_HSHOE_SCALE = 1e-1
DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_CHECKPOINT_EPOCHS = (500, 1000)

DEFAULT_VARIANTS = ("bnn", "hs", "sou_sgl")


def set_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy, Python and PyTorch."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist_cached(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached MNIST arrays from NPZ."""
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.int64)
    y = np.asarray(d["y"])
    if X.shape != (70000, 784) or y.shape != (70000,):
        raise ValueError(f"Unexpected shapes: X={X.shape}, y={y.shape}")
    return X, y


def split_openml_order(X: np.ndarray, y: np.ndarray, n_test: int = 10000):
    """Split OpenML-ordered MNIST into first 60k train and last 10k test."""
    n_total = X.shape[0]
    n_train = n_total - n_test
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def make_torch_mnist(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, int]:
    """Convert cached MNIST arrays into normalized tensors and indexed labels."""
    X_train_np = (X_train_np / 255.0).astype(np.float32)
    X_test_np = (X_test_np / 255.0).astype(np.float32)

    classes, y_train_index = np.unique(y_train_np.astype(int), return_inverse=True)
    n_classes = len(classes)

    class_to_index = {int(c): i for i, c in enumerate(classes)}
    y_test_index = np.array([class_to_index[int(v)] for v in y_test_np.astype(int)], dtype=np.int64)

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_index, dtype=torch.long)

    return X_train_t, y_train_t, X_test_t, y_test_index, n_classes


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
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    return obj


def variant_to_model_and_gamma(variant: str) -> Tuple[str, float | None]:
    """Map variant name to internal model kind and gamma_local value."""
    v = variant.lower()
    if v == "bnn":
        return "bnn", None
    if v == "hs":
        return "sou", 1.0
    if v in {"sou_sgl", "sou-sgl", "sousgl"}:
        return "sou", 1e-9
    raise ValueError(f"Unknown variant: {variant}. Expected one of {DEFAULT_VARIANTS}.")


def _fmt_lr(step_size: float) -> str:
    """Stable compact formatting for learning-rate token in filenames."""
    return f"{float(step_size):.2e}"


def _variant_tag(
    *,
    variant: str,
    num_layers: int,
    num_nodes: int,
    hshoe_scale: float,
    seed: int,
    num_epochs: int,
    batch_size: int,
    step_size: float,
) -> str:
    return (
        f"mnist_bnn_variant={variant}"
        f"_layers={num_layers}"
        f"_nodes={num_nodes}"
        f"_hs={hshoe_scale}"
        f"_ep={num_epochs}"
        f"_bs={batch_size}"
        f"_lr={_fmt_lr(step_size)}"
        f"_seed={seed}"
    )


def _combo_tag(
    *,
    num_layers: int,
    num_nodes: int,
    hshoe_scale: float,
    seed: int,
    num_epochs: int,
    batch_size: int,
    step_size: float,
) -> str:
    return (
        f"mnist_bnn_combo"
        f"_layers={num_layers}"
        f"_nodes={num_nodes}"
        f"_hs={hshoe_scale}"
        f"_ep={num_epochs}"
        f"_bs={batch_size}"
        f"_lr={_fmt_lr(step_size)}"
        f"_seed={seed}"
    )


def run_one_variant(
    *,
    variant: str,
    seed: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test_index: np.ndarray,
    n_classes: int,
    num_epochs: int,
    num_layers: int,
    num_nodes: int,
    hshoe_scale: float,
    checkpoint_epochs: List[int],
    batch_size: int,
    step_size: float,
    warmup_epochs: int,
    verbose: bool,
    device: str | None,
) -> Dict[str, Any]:
    """Train one variant and return a serializable result dictionary."""
    set_seed(seed)

    model_name, gamma = variant_to_model_and_gamma(variant)

    base_config: Dict[str, Any] = {
        "num_epochs": int(num_epochs),
        "batch_size": int(batch_size),
        "verbose": bool(verbose),
        "num_nodes": [int(num_nodes) for _ in range(int(num_layers))],
        "num_layers": int(num_layers),
        "ip_dim": int(X_train.shape[1]),
        "op_dim": int(n_classes),
        "step_size": float(step_size),
        "hshoe_scale": [float(hshoe_scale) for _ in range(int(num_layers))],
        "activation_type": "relu",
        "warmup_epochs": int(warmup_epochs),
    }

    # Only SoU model uses gamma_local
    if model_name == "sou":
        base_config["gamma_local"] = [float(gamma) for _ in range(int(num_layers))]

    torch_device = None
    if device is not None:
        torch_device = torch.device(device)

    model = SoUBnnClassificationHistory(config=base_config, model=model_name, device=torch_device)

    set_seed(seed)
    model.fit(
        X_train,
        y_train,
        X_test=X_test,
        y_test=y_test_index,
        checkpoint_epochs=checkpoint_epochs,
    )

    out = {
        "variant": variant,
        "model": model_name,
        "seed": int(seed),
        "config": base_config,
        "checkpoint_epochs": list(map(int, checkpoint_epochs)),
        "history": to_serializable(getattr(model, "history", None)),
        # Keep an in-memory copy of last state for downstream serialization.
        "last_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run MNIST BNN variants and save training histories.")
    p.add_argument("--seed", type=int, default=1, help="Global random seed.")
    p.add_argument("--mnist-cache", type=str, default=MNIST_CACHE_DEFAULT, help="Cached MNIST NPZ path.")
    p.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT, help="Output directory for NPZ/JSON artifacts.")

    p.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS, help="Training epochs.")
    p.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Number of hidden layers.")
    p.add_argument("--num-nodes", type=int, default=DEFAULT_NUM_NODES, help="Nodes per hidden layer.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    p.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE, help="Optimizer step size.")
    p.add_argument("--hshoe-scale", type=float, default=DEFAULT_HSHOE_SCALE, help="Horseshoe scale.")
    p.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS, help="Warmup epochs.")

    p.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_EPOCHS),
        help="Epochs at which the model records checkpoints in the history.",
    )

    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=f"Variants to run. Default: {DEFAULT_VARIANTS}",
    )

    p.add_argument("--device", type=str, default=None, help="Torch device, e.g. 'cuda', 'cuda:0', or 'cpu'.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logs during fitting.")
    args = p.parse_args()

    mnist_cache = Path(args.mnist_cache)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_mnist_cached(mnist_cache)
    X_train_np, y_train_np, X_test_np, y_test_np = split_openml_order(X, y)

    X_train, y_train, X_test, y_test_index, n_classes = make_torch_mnist(
        X_train_np, y_train_np, X_test_np, y_test_np
    )

    results = []
    for variant in args.variants:
        res = run_one_variant(
            variant=str(variant),
            seed=int(args.seed),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test_index=y_test_index,
            n_classes=n_classes,
            num_epochs=int(args.num_epochs),
            num_layers=int(args.num_layers),
            num_nodes=int(args.num_nodes),
            hshoe_scale=float(args.hshoe_scale),
            checkpoint_epochs=list(map(int, args.checkpoint_epochs)),
            batch_size=int(args.batch_size),
            step_size=float(args.step_size),
            warmup_epochs=int(args.warmup_epochs),
            verbose=bool(args.verbose),
            device=args.device,
        )
        results.append(res)

        tag = _variant_tag(
            variant=str(variant),
            num_layers=int(args.num_layers),
            num_nodes=int(args.num_nodes),
            hshoe_scale=float(args.hshoe_scale),
            seed=int(args.seed),
            num_epochs=int(args.num_epochs),
            batch_size=int(args.batch_size),
            step_size=float(args.step_size),
        )
        npz_path = outdir / f"{tag}.npz"
        json_path = outdir / f"{tag}.json"
        state_path = outdir / f"{tag}_last_state.pt"

        np.savez_compressed(npz_path, result=res)

        # Save last state dict as a dedicated torch artifact.
        torch.save(res["last_state_dict"], state_path)

        # JSON does not store tensors; drop state dict before writing.
        res_json = dict(res)
        res_json.pop("last_state_dict", None)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res_json, f, indent=2)

        print(f"Saved: {npz_path}")
        print(f"Saved: {json_path}")
        print(f"Saved: {state_path}")

    # Combined file
    combo_tag = _combo_tag(
        num_layers=int(args.num_layers),
        num_nodes=int(args.num_nodes),
        hshoe_scale=float(args.hshoe_scale),
        seed=int(args.seed),
        num_epochs=int(args.num_epochs),
        batch_size=int(args.batch_size),
        step_size=float(args.step_size),
    )
    combo_path = outdir / f"{combo_tag}.npz"
    np.savez_compressed(combo_path, results=np.array(results, dtype=object))
    print(f"Saved combined: {combo_path}")


if __name__ == "__main__":
    main()
