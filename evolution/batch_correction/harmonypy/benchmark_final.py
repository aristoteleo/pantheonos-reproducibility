#!/usr/bin/env python
"""Benchmark all Harmony variants on TMA and Pancreas datasets."""

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


def run_benchmark(variants, X, batch_labels, celltype_labels, meta_data, nclust=50):
    print(f"\n  {'Variant':<20} {'Mixing':>8} {'Bio':>8} {'Speed':>8} {'Time(s)':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, path in variants.items():
        try:
            module = load_module(f"harmony_{name}_{id(X)}", path)
            start = time.time()
            hm = module.run_harmony(
                X.copy(), meta_data, vars_use="batch",
                nclust=nclust, max_iter_harmony=10, random_state=42, verbose=False,
            )
            elapsed = time.time() - start
            X_corr = hm.Z_corr

            corr_mag = np.abs(X - X_corr).mean()
            if corr_mag < 0.01:
                print(f"  {name:<20} FAIL: no correction")
                continue

            mixing = compute_batch_mixing_score(X_corr, batch_labels)
            bio = compute_bio_conservation_score(X_corr, celltype_labels)
            speed = 1.0 / (1 + elapsed)
            print(f"  {name:<20} {mixing:>8.4f} {bio:>8.4f} {speed:>8.4f} {elapsed:>8.2f}")
        except Exception as e:
            print(f"  {name:<20} ERROR: {e}")


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / "data"
    results_dir = base_dir / "results"
    results_claude_dir = base_dir / "results_claude_model"

    variants = {}
    variant_files = {
        "original":          base_dir / "harmony.py",
        "harmony_272":       results_dir / "harmony_272.py",
        "best_mixing":       results_dir / "harmony_best_mixing.py",
        "optimized(sonnet)": results_dir / "harmony_optimized.py",
        "optimized(opus)":   results_claude_dir / "harmony_optimized.py",
        "claude":            results_dir / "harmony_claude.py",
    }
    for name, path in variant_files.items():
        if path.exists():
            variants[name] = path
        else:
            print(f"  Skipping {name}: not found")

    # === TMA TEST ===
    print("=" * 70)
    print("  TMA TEST (8000 cells, 2 batches, 12 cell types)")
    print("=" * 70)
    df = pd.read_csv(data_dir / "tma_8000_test.csv")
    X = df.iloc[:, :30].values.astype(np.float32)
    run_benchmark(variants, X, df["donor"].values, df["celltype"].values,
                  pd.DataFrame({"batch": df["donor"].values}), nclust=50)

    # === Pancreas ===
    print("\n" + "=" * 70)
    print("  Pancreas (16382 cells, 9 batches, 14 cell types)")
    print("=" * 70)
    df = pd.read_csv(data_dir / "pancreas.csv")
    X = df.iloc[:, :50].values.astype(np.float32)
    run_benchmark(variants, X, df["batch"].values, df["celltype"].values,
                  pd.DataFrame({"batch": df["batch"].values}), nclust=100)

    print()


if __name__ == "__main__":
    main()
