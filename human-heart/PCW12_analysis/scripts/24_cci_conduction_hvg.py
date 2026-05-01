"""
Spatial CCI for the 3 conduction-cell unordered pairs (6 ordered) on the same
HVG ∪ disease imputed AnnData.

Pairs (both directions):
  CoreConductionCells <-> ACM
  CoreConductionCells <-> TzConductionCells
  CoreConductionCells <-> NC

Inputs:
  PCW12_analysis/data/adata_cci_ready_hvg.h5ad
  PCW12_analysis/data/spateo_lr_db/lr_db_human.csv

Outputs:
  PCW12_analysis/results/cci_hvg/per_pair/<sender>__<receiver>.csv  (overwrite/new)
  PCW12_analysis/results/cci_hvg/conduction_top3_per_pair.csv
  PCW12_analysis/results/cci_hvg/conduction_loop_status.csv

Note: NUM_PERM=50 (vs 10 in 22_*.py) since populations are small (38–56 cells).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

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
SUMMARY_CSV = OUT_DIR / "conduction_top3_per_pair.csv"
LOG_CSV = OUT_DIR / "conduction_loop_status.csv"

GROUP_KEY = "mapped_coarse_celltype"
NUM_PERM = 50
TOP_PER_PAIR = 5

UNORDERED_PAIRS = [
    ("CoreConductionCells", "ACM"),
    ("CoreConductionCells", "TzConductionCells"),
    ("CoreConductionCells", "NC"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"Loading {IN_H5AD.name}")
    adata = sc.read_h5ad(IN_H5AD)
    log(f"  shape={adata.shape}")
    counts = adata.obs[GROUP_KEY].value_counts()
    for a, b in UNORDERED_PAIRS:
        log(f"  {a} n={int(counts.get(a,0))}, {b} n={int(counts.get(b,0))}")

    ordered = [(s, r) for a, b in UNORDERED_PAIRS for (s, r) in [(a, b), (b, a)]]
    summary_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    t_start = time.time()

    for i, (s, r) in enumerate(ordered, 1):
        tag = f"{s}__{r}"
        log(f"[{i}/{len(ordered)}] {s} -> {r}")
        t0 = time.time()
        try:
            st.tl.prepare_cci_cellpair_adata(
                adata, sender_group=s, receiver_group=r,
                group=GROUP_KEY, all_cell_pair=True,
            )
            res = st.tl.find_cci_two_group(
                adata,
                path=str(DB_DIR) + "/",
                species="human",
                group=GROUP_KEY,
                sender_group=s, receiver_group=r,
                filter_lr="outer",
                layer="thresh",
                min_pairs=0, min_pairs_ratio=0,
                top=20, num=NUM_PERM,
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
            log(f"  EMPTY ({dt:.1f}s)")
            status_rows.append({
                "sender": s, "receiver": r, "status": "empty",
                "n_lr": 0, "n_sig": 0, "elapsed_s": dt, "error": "",
            })
            continue

        lr_df = res["lr_pair"].copy()
        lr_df["sender"] = s
        lr_df["receiver"] = r
        lr_df["sr_pair"] = tag
        lr_df.to_csv(PER_PAIR_DIR / f"{tag}.csv", index=False)

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

    log(f"Loop done in {(time.time()-t_start)/60:.1f} min")
    pd.DataFrame(status_rows).to_csv(LOG_CSV, index=False)
    log(f"Wrote {LOG_CSV.name}")

    if summary_rows:
        master = pd.concat(summary_rows, ignore_index=True)
        master.to_csv(SUMMARY_CSV, index=False)
        log(f"Wrote {SUMMARY_CSV.name} — {len(master)} rows from {len(summary_rows)} pairs")
        sort_col = "lr_co_exp_ratio" if "lr_co_exp_ratio" in master.columns else "lr_co_exp_num"
        cols = [c for c in ["sender", "receiver", "from", "to",
                            "lr_co_exp_num", "lr_co_exp_ratio", "is_significant"]
                if c in master.columns]
        log("Top rows by co-expression ratio:")
        print(master.sort_values(sort_col, ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
