#!/usr/bin/env python
"""Benchmark Harmony variants on the pancreas dataset (16k cells, 9 batches, 14 cell types)."""

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import torch
torch.backends.mps.is_available = lambda: False


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compute_batch_mixing_score(X, batch_labels, k=50):
    n_cells = X.shape[0]
    unique_batches = np.unique(batch_labels)
    expected_props = np.array([np.sum(batch_labels == b) / n_cells for b in unique_batches])
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_cells), algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    mixing_scores = []
    for i in range(n_cells):
        neighbor_batches = batch_labels[indices[i, 1:]]
        observed_props = np.array([np.sum(neighbor_batches == b) / k for b in unique_batches])
        score = 1 - np.sqrt(np.mean((observed_props - expected_props) ** 2))
        mixing_scores.append(max(0, score))
    return np.mean(mixing_scores)


def compute_bio_conservation_score(X, labels):
    try:
        if len(np.unique(labels)) > 1:
            return (silhouette_score(X, labels) + 1) / 2
        return 0.5
    except Exception:
        return 0.5


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / "data"
    results_dir = base_dir / "results"

    variants = {
        "original":       base_dir / "harmony.py",
        "harmony_272":    results_dir / "harmony_272.py",
        "best_mixing":    results_dir / "harmony_best_mixing.py",
        "optimized":      results_dir / "harmony_optimized.py",
        "claude":         results_dir / "harmony_claude.py",
    }

    for name, path in list(variants.items()):
        if not path.exists():
            print(f"  WARNING: {name} not found, skipping")
            del variants[name]

    # Load pancreas data
    print("Loading pancreas dataset...")
    df = pd.read_csv(data_dir / "pancreas.csv")
    X = df.iloc[:, :50].values.astype(np.float32)  # 50 PCs
    batch_labels = df["batch"].values
    celltype_labels = df["celltype"].values
    meta_data = pd.DataFrame({"batch": batch_labels})

    print(f"  {X.shape[0]} cells x {X.shape[1]} PCs")
    print(f"  {len(np.unique(batch_labels))} batches: {np.unique(batch_labels).tolist()}")
    print(f"  {len(np.unique(celltype_labels))} cell types")

    # Uncorrected baseline
    print(f"\nComputing uncorrected baseline metrics...")
    mix_uncorr = compute_batch_mixing_score(X, batch_labels)
    bio_uncorr = compute_bio_conservation_score(X, celltype_labels)

    print(f"\n{'='*70}")
    print(f"  Pancreas Benchmark (9 batches, 14 cell types)")
    print(f"{'='*70}")

    print(f"\n  {'Variant':<18} {'Mixing':>8} {'Bio':>8} {'Speed':>8} {'Time(s)':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'uncorrected':<18} {mix_uncorr:>8.4f} {bio_uncorr:>8.4f} {'—':>8} {'—':>8}")

    for name, path in variants.items():
        try:
            module = load_module(f"harmony_pancreas_{name}", path)

            start = time.time()
            hm = module.run_harmony(
                X.copy(), meta_data, vars_use="batch",
                nclust=100, max_iter_harmony=10, random_state=42, verbose=False,
            )
            elapsed = time.time() - start

            X_corr = hm.Z_corr

            corr_mag = np.abs(X - X_corr).mean()
            if corr_mag < 0.01:
                print(f"  {name:<18} FAIL: no correction (mag={corr_mag:.6f})")
                continue

            mixing = compute_batch_mixing_score(X_corr, batch_labels)
            bio = compute_bio_conservation_score(X_corr, celltype_labels)
            speed = 1.0 / (1 + elapsed)

            print(f"  {name:<18} {mixing:>8.4f} {bio:>8.4f} {speed:>8.4f} {elapsed:>8.2f}")

        except Exception as e:
            print(f"  {name:<18} ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
