"""
Spatial CCI loop — Step 2 of 3.

Run `find_cci_two_group` on every ordered pair of the top-8 cell types
(8 × 7 = 56 pairs) using num=10 permutations for exploration.

Per-pair tables:
  PCW12_analysis/results/cci/per_pair/<sender>__<receiver>.csv

Master summary (top-3 LR pairs per ordered cell-type pair):
  PCW12_analysis/results/cci/all_lr_top3_per_pair.csv

NOTE: With only 221 disease genes, the LR universe collapses to ~11 pairs
(mainly NODAL → ACVR/CFC1). The loop still runs the full 56 ordered pairs;
empty / failing pairs are recorded.
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
IN_H5AD = ROOT / "PCW12_analysis/data/adata_cci_ready.h5ad"
DB_DIR = ROOT / "PCW12_analysis/data/spateo_lr_db"
OUT_DIR = ROOT / "PCW12_analysis/results/cci"
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
    assert "thresh" in adata.layers, "missing 'thresh' layer; rerun 15_spatial_cci_prep.py"
    assert "connectivities" in adata.obsp, "missing 'connectivities'"
    assert "__type" in adata.uns, "missing uns['__type']"

    # Top-8 cell types from prep
    if "cci_top_celltypes" in adata.uns:
        top_ct = list(adata.uns["cci_top_celltypes"])
    else:
        top_ct = adata.obs[GROUP_KEY].value_counts().head(8).index.tolist()
    log(f"Top-8 cell types: {top_ct}")
    log(f"Pairs to evaluate: {len(top_ct)*(len(top_ct)-1)} ordered")

    summary_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []

    pair_idx = 0
    n_total = len(top_ct) * (len(top_ct) - 1)
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
                # prepare_cci_cellpair_adata sets up the cellpair label
                st.tl.prepare_cci_cellpair_adata(
                    adata,
                    sender_group=s,
                    receiver_group=r,
                    group=GROUP_KEY,
                    all_cell_pair=True,
                )
                res = st.tl.find_cci_two_group(
                    adata,
                    path=str(DB_DIR) + "/",  # spateo does path + "lr_db_human.csv"
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
                    "n_lr": 0, "elapsed_s": dt, "error": repr(e)[:200],
                })
                continue

            dt = time.time() - t0
            if res is None or "lr_pair" not in res or res["lr_pair"] is None:
                log(f"  EMPTY result ({dt:.1f}s)")
                status_rows.append({
                    "sender": s, "receiver": r, "status": "empty",
                    "n_lr": 0, "elapsed_s": dt, "error": "",
                })
                continue

            lr_df = res["lr_pair"].copy()
            lr_df["sender"] = s
            lr_df["receiver"] = r
            lr_df["sr_pair"] = tag

            # Save full per-pair CSV
            per_pair_csv = PER_PAIR_DIR / f"{tag}.csv"
            lr_df.to_csv(per_pair_csv, index=False)

            # Top-3 by lr_co_exp_ratio (fallback to lr_co_exp_num)
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

    # Write status log
    pd.DataFrame(status_rows).to_csv(LOG_CSV, index=False)
    log(f"Wrote {LOG_CSV.name}")

    # Write master summary
    if summary_rows:
        master = pd.concat(summary_rows, ignore_index=True)
        master.to_csv(SUMMARY_CSV, index=False)
        log(f"Wrote {SUMMARY_CSV.name} — {len(master)} rows from {len(summary_rows)} pairs")
        # Print preview
        cols = [c for c in ["sender", "receiver", "from", "to",
                            "lr_co_exp_num", "lr_co_exp_ratio", "is_significant"]
                if c in master.columns]
        log("Top 10 rows by co-expression ratio:")
        print(master.sort_values(
            "lr_co_exp_ratio" if "lr_co_exp_ratio" in master.columns else "lr_co_exp_num",
            ascending=False)[cols].head(10).to_string(index=False))
    else:
        log("WARNING: no pairs returned any LR rows.")


if __name__ == "__main__":
    main()
