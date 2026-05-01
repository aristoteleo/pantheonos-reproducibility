"""
Spatial CCI loop (v2) — same recipe as 16_cci_loop.py but on the HVG ∪ disease
imputed AnnData. With ~5000 imputed genes the LR universe is much richer,
so we expect many more populated/significant pairs than the 5/56 from the
prior run.

Inputs:
  PCW12_analysis/data/adata_cci_ready_hvg.h5ad
  PCW12_analysis/data/spateo_lr_db/lr_db_human.csv

Outputs:
  PCW12_analysis/results/cci_hvg/per_pair/<sender>__<receiver>.csv
  PCW12_analysis/results/cci_hvg/all_lr_top3_per_pair.csv
  PCW12_analysis/results/cci_hvg/loop_status.csv
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import spateo as st

warnings.filterwarnings("ignore")

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
IN_H5AD = ROOT / "PCW12_analysis/data/adata_cci_ready_hvg.h5ad"
DB_DIR = ROOT / "PCW12_analysis/data/spateo_lr_db"
OUT_DIR = ROOT / "PCW12_analysis/results/cci_hvg"
PER_PAIR_DIR = OUT_DIR / "per_pair"
PER_PAIR_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "all_lr_top3_per_pair.csv"
LOG_CSV = OUT_DIR / "loop_status.csv"

GROUP_KEY = "mapped_coarse_celltype"
NUM_PERM = 10
TOP_PER_PAIR = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {IN_H5AD.name}")
    adata = sc.read_h5ad(IN_H5AD)
    log(f"  shape={adata.shape}, layers={list(adata.layers.keys())}")
    assert "thresh" in adata.layers, "missing 'thresh' layer; rerun 21_spatial_cci_prep_hvg.py"
    assert "connectivities" in adata.obsp, "missing 'connectivities'"
    assert "__type" in adata.uns, "missing uns['__type']"

    if "cci_top_celltypes" in adata.uns:
        top_ct = list(adata.uns["cci_top_celltypes"])
    else:
        top_ct = adata.obs[GROUP_KEY].value_counts().head(8).index.tolist()
    log(f"Top-8 cell types: {top_ct}")
    n_total = len(top_ct) * (len(top_ct) - 1)
    log(f"Pairs to evaluate: {n_total} ordered")

    summary_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    pair_idx = 0
    t_start = time.time()
    for s in top_ct:
        for r in top_ct:
            if s == r:
                continue
            pair_idx += 1
            tag = f"{s}__{r}"
            log(f"[{pair_idx}/{n_total}] {s} -> {r}")

            t0 = time.time()
            try:
                st.tl.prepare_cci_cellpair_adata(
                    adata,
                    sender_group=s,
                    receiver_group=r,
                    group=GROUP_KEY,
                    all_cell_pair=True,
                )
                res = st.tl.find_cci_two_group(
                    adata,
                    path=str(DB_DIR) + "/",
                    species="human",
                    group=GROUP_KEY,
                    sender_group=s,
                    receiver_group=r,
                    filter_lr="outer",
                    layer="thresh",
                    min_pairs=0,
                    min_pairs_ratio=0,
                    top=20,
                    num=NUM_PERM,
                )
            except Exception as e:
                dt = time.time() - t0
                log(f"  EXCEPTION after {dt:.1f}s: {e!r}")
                status_rows.append({
                    "sender": s, "receiver": r, "status": "exception",
                    "n_lr": 0, "n_sig": 0, "elapsed_s": dt, "error": repr(e)[:200],
                })
                continue

            dt = time.time() - t0
            if res is None or "lr_pair" not in res or res["lr_pair"] is None:
                log(f"  EMPTY result ({dt:.1f}s)")
                status_rows.append({
                    "sender": s, "receiver": r, "status": "empty",
                    "n_lr": 0, "n_sig": 0, "elapsed_s": dt, "error": "",
                })
                continue

            lr_df = res["lr_pair"].copy()
            lr_df["sender"] = s
            lr_df["receiver"] = r
            lr_df["sr_pair"] = tag

            per_pair_csv = PER_PAIR_DIR / f"{tag}.csv"
            lr_df.to_csv(per_pair_csv, index=False)

            sort_col = "lr_co_exp_ratio" if "lr_co_exp_ratio" in lr_df.columns else "lr_co_exp_num"
            top = lr_df.sort_values(sort_col, ascending=False).head(TOP_PER_PAIR)
            summary_rows.append(top)

            n_sig = int((lr_df.get("is_significant", pd.Series(dtype=bool)) == True).sum())
            log(f"  OK {dt:.1f}s — {len(lr_df)} LR rows, top {sort_col}={top[sort_col].iloc[0]:.3f}, sig={n_sig}")
            status_rows.append({
                "sender": s, "receiver": r, "status": "ok",
                "n_lr": len(lr_df), "n_sig": n_sig,
                "elapsed_s": dt, "error": "",
            })

    elapsed = time.time() - t_start
    log(f"Loop done in {elapsed/60:.1f} min")

    pd.DataFrame(status_rows).to_csv(LOG_CSV, index=False)
    log(f"Wrote {LOG_CSV.name}")

    if summary_rows:
        master = pd.concat(summary_rows, ignore_index=True)
        master.to_csv(SUMMARY_CSV, index=False)
        log(f"Wrote {SUMMARY_CSV.name} — {len(master)} rows from {len(summary_rows)} pairs")
        cols = [c for c in ["sender", "receiver", "from", "to",
                            "lr_co_exp_num", "lr_co_exp_ratio", "is_significant"]
                if c in master.columns]
        log("Top 10 rows by co-expression ratio:")
        sort_col = "lr_co_exp_ratio" if "lr_co_exp_ratio" in master.columns else "lr_co_exp_num"
        print(master.sort_values(sort_col, ascending=False)[cols].head(10).to_string(index=False))
    else:
        log("WARNING: no pairs returned any LR rows.")


if __name__ == "__main__":
    main()
