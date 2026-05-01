"""
Partition heart disease genes into:
  - direct: present in MERFISH panel (visualize directly)
  - impute: not in MERFISH but present in scRNA-seq (need MOSCOT imputation)
  - missing: not in either dataset

Also defines a curated highlight set (~30 genes) for which we will render GIFs.

Outputs:
  PCW12_analysis/data/disease_genes_partition.tsv
  PCW12_analysis/data/highlight_genes.tsv
"""
import sys
from pathlib import Path
import pandas as pd
import anndata as ad

ROOT = Path("/Users/weizexu/Projects/pantheonos-reproducibility/human-heart")
SC_PATH = ROOT / "data/all_healthy_RoundedPCW11-13.h5ad"
SP_PATH = ROOT / "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"
TSV     = ROOT / "data/heart_disease_genes.tsv"
OUTDIR  = ROOT / "PCW12_analysis/data"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1. Disease gene table
df = pd.read_csv(TSV, sep="\t")
print(f"Disease gene rows: {len(df)}; unique genes: {df['Gene'].nunique()}; categories: {df['Category'].unique().tolist()}")

# 2. Var names only (cheap)
sc_vars = ad.read_h5ad(SC_PATH, backed='r').var_names
sp_vars = ad.read_h5ad(SP_PATH, backed='r').var_names
print(f"SC genes: {len(sc_vars)}, SP genes: {len(sp_vars)}")

genes = sorted(df["Gene"].unique())
sc_set, sp_set = set(sc_vars), set(sp_vars)

partition = []
for g in genes:
    in_sp = g in sp_set
    in_sc = g in sc_set
    if in_sp:
        cls = "direct"
    elif in_sc:
        cls = "impute"
    else:
        cls = "missing"
    # Aggregate categories/traits
    sub = df[df["Gene"] == g]
    partition.append({
        "Gene": g,
        "in_sc": in_sc,
        "in_sp": in_sp,
        "class": cls,
        "categories": ";".join(sorted(sub["Category"].unique())),
        "traits": ";".join(sorted(sub["Trait"].unique())),
    })
part_df = pd.DataFrame(partition)
print(part_df["class"].value_counts())

out = OUTDIR / "disease_genes_partition.tsv"
part_df.to_csv(out, sep="\t", index=False)
print(f"Wrote {out}")

# 3. Curated highlight gene set: well-known cardiac TFs + structural + key disease drivers
HIGHLIGHTS = [
    # Core cardiac TFs (CHD)
    "GATA4", "GATA6", "NKX2-5", "TBX5", "TBX20", "TBX1",
    "HAND1", "HAND2", "MEF2C", "MEIS2", "PITX2",
    # Sarcomere / cardiomyocyte structural (cardiomyopathy)
    "MYH6", "MYH7", "MYBPC3", "TNNT2", "TNNI3", "TNNC1",
    "ACTC1", "ACTN2", "TPM1", "TTN", "MYL2", "MYL3",
    "DES", "VCL", "PLN",
    # Conduction / ion channel
    "SCN5A", "GJA1",
    # Notch / valve / vascular
    "NOTCH1", "JAG1", "DLL4",
    # Smooth muscle / aortic
    "MYH11", "ACTA2", "FBN1", "ELN",
    # Other PCGC frequent hits
    "CHD7", "KMT2D", "CHD4",
]
hi_df = part_df[part_df["Gene"].isin(HIGHLIGHTS)].copy()
hi_df["highlight"] = True
hi_path = OUTDIR / "highlight_genes.tsv"
hi_df.to_csv(hi_path, sep="\t", index=False)
print(f"Highlight genes saved: {hi_path}")
print(hi_df.groupby("class")["Gene"].apply(list))
