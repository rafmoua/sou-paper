from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml


def download_mnist_openml(out_path: Path, force: bool = False) -> None:
    """Download MNIST from OpenML and store a local compressed NPZ cache."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"File already exists: {out_path}")
        print("Use --force to re-download.")
        return

    print("Downloading MNIST from OpenML...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    if X.shape != (70000, 784):
        raise ValueError(f"Unexpected X shape: {X.shape} (expected (70000, 784))")
    if y.shape != (70000,):
        raise ValueError(f"Unexpected y shape: {y.shape} (expected (70000,))")

    X = np.asarray(X, dtype=np.int64)

    y = np.asarray(y)

    meta = {
        "source": "openml",
        "openml_name": "mnist_784",
        "openml_version": 1,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "X_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
    }

    print(f"Saving to: {out_path}")
    np.savez_compressed(out_path, X=X, y=y, meta=meta)

    print("Saved MNIST cache.")
    print(f"X: shape={X.shape}, dtype={X.dtype}")
    print(f"y: shape={y.shape}, dtype={y.dtype}")


def main() -> None:
    args = parse_args()
    download_mnist_openml(Path(args.out), force=args.force)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for MNIST download/cache."""
    p = argparse.ArgumentParser(description="Download and cache MNIST from OpenML.")
    p.add_argument(
        "--out",
        type=str,
        default="data/raw/mnist_784_openml_v1.npz",
        help="Output path for cached MNIST .npz",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if the file already exists.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
