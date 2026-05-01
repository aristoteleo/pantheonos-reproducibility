# PantheonOS Reproducibility

This repository contains reproduction materials (scripts, intermediate results, reports) for the figures in the **PantheonOS** paper.

The actual reasoning trajectories the agent took to produce each figure are recorded as **replayable bundles** hosted on Hugging Face. Click any "▶ Replay" link below to step through the exact agent conversation, tool calls, and intermediate outputs in your browser via the public PantheonOS replay viewer.

Raw `.h5ad` inputs are not redistributed here. They come from two upstream studies — see the **Data sources** section below for the original data and access instructions.

---

## Figures → Trajectories / Reports

### Figure 2 — Immune Oncology Gene Panel Design

A 1000-plex human immune-oncology gene panel designed by multi-agent collaboration. The team downloads CELLxGENE Census data, runs HVG/DE/RF/scGeneFit/SpaPROS gene selection algorithms, benchmarks panel quality, and produces a final annotated panel with LaTeX report.

- ▶ [Replay trajectory](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fimmune_oncology_gene_panel_design.zip)

### Figure 3 — Evolution (algorithm self-improvement)

Pantheon's evolution system iteratively improves analysis algorithms — both batch-correction methods and an RL-based gene-panel selector.

**Batch correction** (BBKNN / Harmonypy / Scanorama):

| Method | HTML report | Code & run artifacts |
| --- | --- | --- |
| BBKNN | 📄 [Report](https://pantheonos.stanford.edu/evolve/bbknn.html) | [`evolution/batch_correction/bbknn/`](evolution/batch_correction/bbknn/) |
| Harmonypy | 📄 [Report](https://pantheonos.stanford.edu/evolve/harmonypy.html) | [`evolution/batch_correction/harmonypy/`](evolution/batch_correction/harmonypy/) |
| Scanorama | 📄 [Report](https://pantheonos.stanford.edu/evolve/scanorama.html) | [`evolution/batch_correction/scanorama/`](evolution/batch_correction/scanorama/) |

**RL gene-panel selection**: [`evolution/gene_panel/`](evolution/gene_panel/) — exploration- and optimization-mode runs of an RL panel selector evolved with the same `pantheon.evolution` engine.

See [`evolution/README.md`](evolution/README.md) for the layout of run scripts, evaluators, JSON trajectory data, and evolved variants.

### Figure 4 — *(coming soon)*

### Figure 5 — 3D Human Fetal Heart Multi-Omic Analysis

A four-step analysis workflow on the PCW12 human fetal heart MERFISH + scRNA + snATAC dataset.

| Step | Trajectory |
| --- | --- |
| 5.1 Heart MERFISH data overview | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fheart_merfish_data_overview.zip) |
| 5.2 3D human fetal heart disease gene pattern | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fhuman_heart_3d_disease_gene_pattern.zip) |
| 5.3 Spatial ligand-receptor disease analysis | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fspatial_ligand_receptor_disease.zip) |
| 5.4 ATAC spatial mapping via MOSCOT | ▶ [Replay](https://pantheon-ui.aristoteleo.com/#/replay?url=https%3A%2F%2Fhuggingface.co%2Fdatasets%2FNaNg%2FPantheon-Agent-Trajectory%2Fresolve%2Fmain%2Fatac_spatial_mapping_moscot.zip) |

Local working files for Figure 5 live under [`human-heart/`](human-heart/) — `PCW7_analysis/` and `PCW12_analysis/` contain scripts, figures, and reports. Raw `.h5ad` inputs come from the upstream MERFISH+ and human fetal multi-omic studies (see [Data sources](#data-sources)).

### Figure 6 — *(coming soon)*

---

## Repository layout

```
evolution/                  # Figure 3 — algorithm self-improvement runs
├── batch_correction/       # BBKNN / Harmonypy / Scanorama
└── gene_panel/             # RL gene-panel selector

human-heart/                # Figure 5 working directory
├── PCW7_analysis/          # Earlier developmental stage (scripts, results)
├── PCW12_analysis/         # Main figure-5 analyses
│   ├── scripts/            # Reproducible Python scripts
│   ├── figures/            # Final figure PNGs/PDFs
│   ├── REPORT.md           # Auto-generated analysis report
│   └── REPORT_ATAC.md      # ATAC sub-analysis report
├── data/                   # Raw .h5ad inputs (upstream papers, gitignored)
└── references/

mouse-embryo-e6/            # In progress
```

The full Pantheon project state (chat memory, agent runtime, virtual envs) is gitignored — see `.gitignore`. To re-run any analysis end-to-end, replay the corresponding trajectory in PantheonOS Desktop or via the web replayer.

---

## Data sources

The Figure 5 raw `.h5ad` inputs come from two upstream studies, not redistributed in this repository:

| Modality | Preprint | Used for |
| --- | --- | --- |
| Single-cell multiome (snRNA + snATAC) of the human fetal heart | [medRxiv 2024.11.20.24317557](https://www.medrxiv.org/content/10.1101/2024.11.20.24317557v2) | scRNA / snATAC inputs, CHD enhancer–gene mapping |
| 3D MERFISH whole-organ spatial transcriptomics | [bioRxiv 2025.11.02.686137](https://www.biorxiv.org/content/10.1101/2025.11.02.686137v1) | PCW12 3D MERFISH cell × gene tensor |

Refer to each preprint's *Data Availability* section for the canonical accession links.

## Other hosted assets

| Asset | Where |
| --- | --- |
| Replayable agent trajectories | [`NaNg/Pantheon-Agent-Trajectory`](https://huggingface.co/datasets/NaNg/Pantheon-Agent-Trajectory) on Hugging Face |
| Evolution HTML reports | https://pantheonos.stanford.edu/evolve/ |

---

## License

See [LICENSE](LICENSE).
