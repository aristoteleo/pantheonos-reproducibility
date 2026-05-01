"""Quick metadata inspection of both datasets."""
import anndata as ad
import json
import sys

results = {}

for name, path in [
    ("scrna", "data/all_healthy_RoundedPCW11-13.h5ad"),
    ("merfish", "data/full_heart_final_aug2025_update_downsampled_100k.h5ad"),
]:
    print(f"\n{'='*60}\n{name}: {path}\n{'='*60}", flush=True)
    a = ad.read_h5ad(path)
    info = {
        "shape": list(a.shape),
        "obs_columns": list(a.obs.columns),
        "obs_dtypes": {c: str(a.obs[c].dtype) for c in a.obs.columns},
        "var_columns": list(a.var.columns),
        "obsm_keys": list(a.obsm.keys()),
        "obsm_shapes": {k: list(a.obsm[k].shape) for k in a.obsm.keys()},
        "uns_keys": list(a.uns.keys()),
        "layers": list(a.layers.keys()),
        "X_dtype": str(a.X.dtype),
        "X_min_max_sample": [float(a.X[:1000].min()), float(a.X[:1000].max())] if a.X.shape[0] >= 1000 else None,
    }
    # Sample values for categorical columns
    cat_samples = {}
    for c in a.obs.columns:
        try:
            uniq = a.obs[c].unique()
            if len(uniq) <= 50:
                cat_samples[c] = sorted([str(x) for x in uniq])
            else:
                cat_samples[c] = f"<{len(uniq)} unique values, sample: {sorted([str(x) for x in uniq[:5]])}>"
        except Exception as e:
            cat_samples[c] = f"<error: {e}>"
    info["obs_value_samples"] = cat_samples
    results[name] = info
    print(json.dumps(info, indent=2, default=str)[:5000], flush=True)

with open("PCW12_analysis/figures/overview/_metadata.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved metadata to PCW12_analysis/figures/overview/_metadata.json")
